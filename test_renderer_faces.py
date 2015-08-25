#!/usr/bin/env python
# encoding: utf-8
"""
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages/')
import time
import math
import unittest
import numpy as np
import unittest
try:
    import matplotlib.pyplot as plt
    import matplotlib
except:
    from dummy import dummy as plt

from opendr.renderer import *
from chumpy import Ch
from chumpy.utils import row, col
from opendr.lighting import *
from opendr.util_tests import get_earthmesh, process
from collections import OrderedDict


    
visualize = True 
    
def getcam():
    from opendr.camera import ProjectPoints

    #w = 256
    #h = 192
    w = 1500
    h = 1500

    #f = np.array([200,200])
    f = np.array([w,h])
    rt = np.zeros(3)
    t = np.zeros(3)
    k = np.zeros(5)
    c = np.array([w/2., h/2.])

    if True:
        ratio = 640 / 256.
        f *= ratio
        c *= ratio
        w *= ratio
        h *= ratio

    pp = ProjectPoints(f=f, rt=rt, t=t, k=k, c=c)
    frustum = {'near': 0.0, 'far': 20.0, 'width': w, 'height': h}

    return pp, frustum


#Writing my own OBJ reader
def load_obj(fname):
  vertices = []
  vt = []
  faces = []
  for line in open(fname,'r'):
    vals = line.split()
    if vals[0] == 'v':
        vertices.append(vals[1:])
    elif vals[0] == 'vt':
        vt.append(vals[1:])
    elif vals[0] == 'f':
        faces.append(vals[1:])
  return vertices, vt, faces

def make_tris(faces):
  quads = np.zeros(4)
  tris = np.zeros((2*len(faces),3))
  idx = 0
  for ii in np.arange(len(faces)):
      tmp = str(faces[ii]).lstrip('[')
      tmp = str(tmp).rstrip(']')
      tmp = str(tmp).split(',')
      quads[0] = np.int32(str(str(tmp[0]).split('/')[0]).lstrip('\'')) 
      quads[1] = np.int32(str(str(tmp[1]).split('/')[0]).lstrip(' \'')) 
      quads[2] = np.int32(str(str(tmp[2]).split('/')[0]).lstrip(' \'')) 
      quads[3] = np.int32(str(str(tmp[3]).split('/')[0]).lstrip(' \'')) 
      tris[idx,0] = quads[0]
      tris[idx,1] = quads[1]
      tris[idx,2] = quads[2]
      idx = idx + 1
      tris[idx,0] = quads[2]
      tris[idx,1] = quads[3]
      tris[idx,2] = quads[0]
      idx = idx + 1
  tris = tris - 1.
  tris = np.int32(tris,dtype='int32')
  return tris

class TestRenderer(unittest.TestCase):

    def load_basics(self):
        np.random.seed(0)
        fname='/Users/mudigonda/Data/faces/fwh/Tester_1/TrainingPose/pose_0.obj'
        from opendr.serialization import load_mesh
        verts, vt, faces = load_obj(fname)
        mesh = load_mesh(fname)
        tris = make_tris(faces)
        mesh.f = tris 
        mesh.vc = np.ones(np.float32(verts).shape)
        camera, frustum = getcam()
        #mesh = get_earthmesh(trans=np.array([0,0,5]), rotation = np.array([0,0,0]))
        
        lighting_3channel = LambertianPointLight(
            f=mesh.f,
            num_verts=len(mesh.v),
            light_pos=np.array([1,1,1]),
            vc=mesh.vc,
            light_color=np.array([1., 1., 1.]))
        lighting_1channel = LambertianPointLight(
            f=mesh.f,
            num_verts=len(mesh.v),
            light_pos=np.array([1,1,1]),
            vc=mesh.vc.mean(axis=1).reshape((-1,1)),
            light_color=np.array([1.]))

        bgcolor = np.array([0.,0.,0.])
        renderers = [
            ColoredRenderer(f=mesh.f, camera=camera, frustum=frustum, bgcolor=bgcolor, num_channels=3),
            #TexturedRenderer(f=mesh.f, camera=camera, frustum=frustum, texture_image=mesh.texture_image, vt=mesh.vt, ft=mesh.ft, bgcolor=bgcolor),
            ColoredRenderer(f=mesh.f, camera=camera, frustum=frustum, bgcolor=bgcolor[0], num_channels=1)]

        lightings = {1: lighting_1channel, 3: lighting_3channel}
        return mesh, lightings, camera, frustum, renderers
        
    def test_pyramids(self):
        """ Test that pyramid construction doesn't crash. No quality testing here. """
        mesh, lightings, camera, frustum, renderers = self.load_basics()
        from opendr.filters import gaussian_pyramid, laplacian_pyramid, GaussPyrDownOne

        camera.v = mesh.v
        for rn in renderers:
            lightings[rn.num_channels].v = camera.v
            rn.vc = lightings[rn.num_channels]
            rn_pyr = gaussian_pyramid(rn, normalization=None, n_levels=2)
            rn_lap = laplacian_pyramid(rn, normalization=None, imshape=rn.shape, as_list=False, n_levels=2)
            rn_gpr = GaussPyrDownOne(im_shape=rn.shape, want_downsampling=True, px=rn)
            for r in [rn_pyr, rn_lap, rn_gpr]:
                _ = r.r

            for r in [rn_pyr, rn_gpr]:
                for ii in range(3):
                    rn.v[:,:] = rn.v[:,:].r + 1e-10
                    import time
                    tm = time.time()
                    _ = r.dr_wrt(rn)
                    #print "trial %d: %.2fS " % (ii, time.time() - tm)
        
    def test_distortion(self):
        mesh, lightings, camera, frustum, renderers = self.load_basics()

        renderer = renderers[1]
        lighting = lightings[renderer.num_channels]
        lighting.light_pos = -lighting.light_pos * 100.

        mesh = get_earthmesh(trans=np.array([0,0,-8]), rotation = np.array([math.pi/2.,0,0]))
        mesh_verts = Ch(mesh.v.flatten())
        renderer.camera = camera
        camera.v = mesh_verts
        lighting.v = mesh_verts
        renderer.vc = lighting
        renderer.camera = camera

        camera.rt = np.array([np.pi, 0, 0])

        # Get pixels and derivatives
        im_original = renderer.r.copy()

        #camera.k = np.zeros(5)
        #camera.k = np.arange(8,0,-1)*.1
        #camera.k = np.array([ 0.00249999,  0.42208098,  0.45360267,  0.06808415, -0.38003062])
        camera.k = np.array([ 5., 25., .3, .4, 1000., 5., 0., 0.])
        im_distorted = renderer.r

        cr = renderer
        cmtx = np.array([
            [cr.camera.f.r[0], 0, cr.camera.c.r[0]],
            [0, cr.camera.f.r[1], cr.camera.c.r[1]],
            [0, 0, 1]
        ])

        import cv2
        im_undistorted = cv2.undistort(im_distorted, cmtx, cr.camera.k.r)

        d1 = (im_original - im_distorted).ravel()
        d2 = (im_original - im_undistorted).ravel()

        d1 = d1[d1 != 0.]
        d2 = d2[d2 != 0.]

        self.assertGreater(np.mean(d1**2) / np.mean(d2**2), 44.)
        self.assertLess(np.mean(d2**2), 0.0016)
        self.assertGreater(np.median(d1**2) / np.median(d2**2), 650)
        self.assertLess(np.median(d2**2), 1.9e-5)


        if visualize:
            import matplotlib.pyplot as plt
            plt.ion()

            matplotlib.rcParams.update({'font.size': 18})
            plt.figure(figsize=(6*3, 2*3))
            plt.subplot(1,4,1)
            plt.imshow(im_original)
            plt.title('original')

            plt.subplot(1,4,2)
            plt.imshow(im_distorted)
            plt.title('distorted')

            plt.subplot(1,4,3)
            plt.imshow(im_undistorted)
            plt.title('undistorted by opencv')

            plt.subplot(1,4,4)
            plt.imshow(im_undistorted - im_original + .5)
            plt.title('diff')

            plt.draw()
            plt.show()






    def test_cam_derivatives(self):
        mesh, lightings, camera, frustum, renderers = self.load_basics()

        camparms = {
            'c': {'mednz' : 2.2e-2, 'meannz': 4.2e-2, 'desc': 'center of proj diff', 'eps0': 4., 'eps1': .1},
            #'f': {'mednz' : 2.5e-2, 'meannz': 6e-2, 'desc': 'focal diff', 'eps0': 100., 'eps1': .1},
            't': {'mednz' : 1.2e-1, 'meannz': 3.0e-1, 'desc': 'trans diff', 'eps0': .25, 'eps1': .1},
            'rt': {'mednz' : 8e-2, 'meannz': 1.8e-1, 'desc': 'rot diff', 'eps0': 0.02, 'eps1': .5},
            'k': {'mednz' : 7e-2, 'meannz': 5.1e-1, 'desc': 'distortion diff', 'eps0': .5, 'eps1': .05}
        }

        for renderer in renderers:

            im_shape = renderer.shape
            lighting = lightings[renderer.num_channels]

            # Render a rotating mesh
            #mesh = get_earthmesh(trans=np.array([0,0,5]), rotation = np.array([math.pi/2.,0,0]))        

            fname='/Users/mudigonda/Data/faces/fwh/Tester_1/TrainingPose/pose_0.obj'
            from opendr.serialization import load_mesh
            verts, vt, faces = load_obj(fname)
            mesh = load_mesh(fname)
            tris = make_tris(faces)
            mesh.f = tris 
            mesh.vc = np.ones(np.float32(verts).shape)
            #STuff from before
            mesh_verts = Ch(mesh.v.flatten())
            camera.v = mesh_verts
            lighting.v = mesh_verts
            renderer.vc = lighting
            renderer.camera = camera


            for atrname, info in camparms.items():

                # Get pixels and derivatives
                r = renderer.r

                atr = lambda : getattr(camera, atrname)
                satr = lambda x : setattr(camera, atrname, x)

                atr_size = atr().size
                dr = renderer.dr_wrt(atr())

                # Establish a random direction
                tmp = np.random.rand(atr().size) - .5
                direction = (tmp / np.linalg.norm(tmp))*info['eps0']
                #direction = np.sin(np.ones(atr_size))*info['eps0']
                #direction = np.zeros(atr_size)
                # try:
                #     direction[4] = 1.
                # except: pass
                #direction *= info['eps0']
                eps = info['eps1']

                # Render going forward in that direction
                satr(atr().r + direction*eps/2.)
                rfwd = renderer.r

                # Render going backward in that direction
                satr(atr().r - direction*eps/1.)
                rbwd = renderer.r

                # Put back
                satr(atr().r + direction*eps/2.)

                # Establish empirical and predicted derivatives
                dr_empirical = (np.asarray(rfwd, np.float64) - np.asarray(rbwd, np.float64)).ravel() / eps
                dr_predicted = dr.dot(col(direction.flatten())).reshape(dr_empirical.shape)

                images = OrderedDict()
                images['shifted %s' % (atrname,)] = np.asarray(rfwd, np.float64)-.5
                images[r'empirical %s' % (atrname,)] = dr_empirical
                images[r'predicted %s' % (atrname,)] = dr_predicted
                images[info['desc']] = dr_predicted - dr_empirical

                nonzero = images[info['desc']][np.nonzero(images[info['desc']]!=0)[0]]

                mederror = np.median(np.abs(nonzero))
                meanerror = np.mean(np.abs(nonzero))
                if visualize:
                    matplotlib.rcParams.update({'font.size': 18})
                    plt.figure(figsize=(6*3, 2*3))
                    plt.title('Test Camera Derivatives')
                    for idx, title in enumerate(images.keys()):
                        plt.subplot(1,len(images.keys()), idx+1)
                        im = process(images[title].reshape(im_shape), vmin=-.5, vmax=.5)
                        plt.title(title)
                        plt.imshow(im)

                    print '%s: median nonzero %.2e' % (atrname, mederror,)
                    print '%s: mean nonzero %.2e' % (atrname, meanerror,)
                    plt.draw()
                    plt.show()

                self.assertLess(meanerror, info['meannz'])
                self.assertLess(mederror, info['mednz'])

        
    def test_vert_derivatives(self):

        mesh, lightings, camera, frustum, renderers = self.load_basics()

        for renderer in renderers:

            lighting = lightings[renderer.num_channels]
            im_shape = renderer.shape

            # Render a rotating mesh
            #mesh = get_earthmesh(trans=np.array([0,0,5]), rotation = np.array([math.pi/2.,0,0]))        
            fname='/Users/mudigonda/Data/faces/fwh/Tester_1/TrainingPose/pose_0.obj'
            from opendr.serialization import load_mesh
            verts, vt, faces = load_obj(fname)
            mesh = load_mesh(fname)
            tris = make_tris(faces)
            mesh.f = tris 
            mesh.vc = np.ones(np.float32(verts).shape)
            mesh_verts = Ch(mesh.v.flatten())  
            camera.set(v=mesh_verts)
            lighting.set(v=mesh_verts)
            renderer.set(camera=camera)
            renderer.set(vc=lighting)

            # Get pixels and derivatives
            r = renderer.r
            dr = renderer.dr_wrt(mesh_verts)
            
            # Establish a random direction
            #direction = (np.random.rand(mesh.v.size).reshape(mesh.v.shape)-.5)*.1 + np.sin(mesh.v*10)*.2
            direction = (np.ones(mesh.v.size).reshape(mesh.v.shape)-.5)*.01 
            direction *= .25
            eps = .02

            # Render going forward in that direction
            mesh_verts = Ch(mesh.v+direction*eps/2.)
            lighting.set(v=mesh_verts)
            renderer.set(v=mesh_verts, vc=lighting)
            rfwd = renderer.r
            
            # Render going backward in that direction
            mesh_verts = Ch(mesh.v-direction*eps/2.)
            lighting.set(v=mesh_verts)
            renderer.set(v=mesh_verts, vc=lighting)
            rbwd = renderer.r

            # Establish empirical and predicted derivatives
            dr_empirical = (np.asarray(rfwd, np.float64) - np.asarray(rbwd, np.float64)).ravel() / eps
            dr_predicted = dr.dot(col(direction.flatten())).reshape(dr_empirical.shape) 
            import IPython; IPython.embed()

            images = OrderedDict()
            images['shifted verts'] = np.asarray(rfwd, np.float64)-.5
            images[r'empirical verts $\left(\frac{dI}{dV}\right)$'] = dr_empirical
            images[r'predicted verts $\left(\frac{dI}{dV}\right)$'] = dr_predicted
            images['difference verts'] = dr_predicted - dr_empirical

            nonzero = images['difference verts'][np.nonzero(images['difference verts']!=0)[0]]

            if visualize:
                matplotlib.rcParams.update({'font.size': 18})
                plt.figure(figsize=(6*3, 2*3))
                plt.title('Test vertices derivatives')
                for idx, title in enumerate(images.keys()):
                    plt.subplot(1,len(images.keys()), idx+1)
                    im = process(images[title].reshape(im_shape), vmin=-.5, vmax=.5)
                    plt.title(title)
                    plt.imshow(im)
                    
                print 'verts: median nonzero %.2e' % (np.median(np.abs(nonzero)),)
                print 'verts: mean nonzero %.2e' % (np.mean(np.abs(nonzero)),)
                plt.draw()
                plt.show()

            self.assertLess(np.mean(np.abs(nonzero)), 7e-2)
            self.assertLess(np.median(np.abs(nonzero)), 4e-2)
            

    def test_lightpos_derivatives(self):
        
        mesh, lightings, camera, frustum, renderers = self.load_basics()
        

        for renderer in renderers:

            im_shape = renderer.shape
            lighting = lightings[renderer.num_channels]

            # Render a rotating mesh
            #mesh = get_earthmesh(trans=np.array([0,0,5]), rotation = np.array([math.pi/2.,0,0]))        
            fname='/Users/mudigonda/Data/faces/fwh/Tester_1/TrainingPose/pose_0.obj'
            from opendr.serialization import load_mesh
            verts, vt, faces = load_obj(fname)
            mesh = load_mesh(fname)
            tris = make_tris(faces)
            mesh.f = tris 
            mesh.vc = np.ones(np.float32(verts).shape)
            mesh_verts = Ch(mesh.v.flatten())
            camera.set(v=mesh_verts)


            # Get predicted derivatives wrt light pos
            light1_pos = Ch(np.array([-1000,-1000,-1000]))
            lighting.set(light_pos=light1_pos, v=mesh_verts)
            renderer.set(vc=lighting, v=mesh_verts)
            
            dr = renderer.dr_wrt(light1_pos).copy()            

            # Establish a random direction for the light
            direction = (np.random.rand(3)-.5)*1000.
            eps = 1.
        
            # Find empirical forward derivatives in that direction
            lighting.set(light_pos = light1_pos.r + direction*eps/2.)
            renderer.set(vc=lighting)
            rfwd = renderer.r
        
            # Find empirical backward derivatives in that direction
            lighting.set(light_pos = light1_pos.r - direction*eps/2.)
            renderer.set(vc=lighting)
            rbwd = renderer.r
        
            # Establish empirical and predicted derivatives
            dr_empirical = (np.asarray(rfwd, np.float64) - np.asarray(rbwd, np.float64)).ravel() / eps
            dr_predicted = dr.dot(col(direction.flatten())).reshape(dr_empirical.shape)

            images = OrderedDict()
            images['shifted lightpos'] = np.asarray(rfwd, np.float64)-.5
            images[r'empirical lightpos $\left(\frac{dI}{dL_p}\right)$'] = dr_empirical
            images[r'predicted lightpos $\left(\frac{dI}{dL_p}\right)$'] = dr_predicted
            images['difference lightpos'] = dr_predicted-dr_empirical

            nonzero = images['difference lightpos'][np.nonzero(images['difference lightpos']!=0)[0]]

            if visualize:
                matplotlib.rcParams.update({'font.size': 18})
                plt.figure(figsize=(6*3, 2*3))
                plt.title('Test light position derivatives')
                for idx, title in enumerate(images.keys()):
                    plt.subplot(1,len(images.keys()), idx+1)
                    im = process(images[title].reshape(im_shape), vmin=-.5, vmax=.5)
                    plt.title(title)
                    plt.imshow(im)
                
                plt.show()
                print 'lightpos: median nonzero %.2e' % (np.median(np.abs(nonzero)),)
                print 'lightpos: mean nonzero %.2e' % (np.mean(np.abs(nonzero)),)
            self.assertLess(np.mean(np.abs(nonzero)), 2.4e-2)
            self.assertLess(np.median(np.abs(nonzero)), 1.2e-2)
            
        
        
    def test_color_derivatives(self):
        
        mesh, lightings, camera, frustum, renderers = self.load_basics()
        
        for renderer in renderers:

            im_shape = renderer.shape
            lighting = lightings[renderer.num_channels]

            # Get pixels and dI/dC
            #mesh = get_earthmesh(trans=np.array([0,0,5]), rotation = np.array([math.pi/2.,0,0]))        
            fname='/Users/mudigonda/Data/faces/fwh/Tester_1/TrainingPose/pose_0.obj'
            from opendr.serialization import load_mesh
            verts, vt, faces = load_obj(fname)
            mesh = load_mesh(fname)
            tris = make_tris(faces)
            mesh.f = tris 
            mesh.vc = np.ones(np.float32(verts).shape)
            mesh_verts = Ch(mesh.v)
            mesh_colors = Ch(mesh.vc)

            camera.set(v=mesh_verts)            

            # import pdb; pdb.set_trace()
            # print '-------------------------------------------'
            #lighting.set(vc=mesh_colors, v=mesh_verts)

            try:
                lighting.vc = mesh_colors[:,:renderer.num_channels]
            except:
                import pdb; pdb.set_trace()
            lighting.v = mesh_verts

            renderer.set(v=mesh_verts, vc=lighting)

            r = renderer.r
            dr = renderer.dr_wrt(mesh_colors).copy()

            # Establish a random direction
            eps = .4
            direction = (np.random.randn(mesh.v.size).reshape(mesh.v.shape)*.1 + np.sin(mesh.v*19)*.1).flatten()

            # Find empirical forward derivatives in that direction
            mesh_colors = Ch(mesh.vc+direction.reshape(mesh.vc.shape)*eps/2.)
            lighting.set(vc=mesh_colors[:,:renderer.num_channels])
            renderer.set(vc=lighting)
            rfwd = renderer.r

            # Find empirical backward derivatives in that direction
            mesh_colors = Ch(mesh.vc-direction.reshape(mesh.vc.shape)*eps/2.)
            lighting.set(vc=mesh_colors[:,:renderer.num_channels])
            renderer.set(vc=lighting)
            rbwd = renderer.r

            dr_empirical = (np.asarray(rfwd, np.float64) - np.asarray(rbwd, np.float64)).ravel() / eps

            try:
                dr_predicted = dr.dot(col(direction.flatten())).reshape(dr_empirical.shape)
            except:
                import pdb; pdb.set_trace()

            images = OrderedDict()
            images['shifted colors'] = np.asarray(rfwd, np.float64)-.5
            images[r'empirical colors $\left(\frac{dI}{dC}\right)$'] = dr_empirical
            images[r'predicted colors $\left(\frac{dI}{dC}\right)$'] = dr_predicted
            images['difference colors'] = dr_predicted-dr_empirical

            nonzero = images['difference colors'][np.nonzero(images['difference colors']!=0)[0]]

            if visualize:
                matplotlib.rcParams.update({'font.size': 18})
                plt.figure(figsize=(6*3, 2*3))
                plt.title('Test Color derivatives') 
                for idx, title in enumerate(images.keys()):
                    plt.subplot(1,len(images.keys()), idx+1)
                    im = process(images[title].reshape(im_shape), vmin=-.5, vmax=.5)
                    plt.title(title)
                    plt.imshow(im)
                    
                plt.show()
                print 'color: median nonzero %.2e' % (np.median(np.abs(nonzero)),)
                print 'color: mean nonzero %.2e' % (np.mean(np.abs(nonzero)),)
            self.assertLess(np.mean(np.abs(nonzero)), 2e-2)
            self.assertLess(np.median(np.abs(nonzero)), 4.5e-3)
                     


def plt_imshow(im):
    #im = process(im, vmin, vmax)    
    result = plt.imshow(im)
    plt.axis('off')
    plt.subplots_adjust(bottom=0.01, top=.99, left=0.01, right=.99)    
    return result


if __name__ == '__main__':
    plt.ion()
    visualize = True
    #unittest.main()
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRenderer)
    import IPython; IPython.embed()
    unittest.TextTestRunner(verbosity=2).run(suite)
    plt.show()

