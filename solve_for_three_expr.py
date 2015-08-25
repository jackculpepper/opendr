#Script to use all of OpenDR's demo to visualize faces
#Here we limit solving expressions to only three of them. More manageable and solveable 

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
    import matplotlib.cm as cm
except:
    from dummy import dummy as plt

from opendr.renderer import *
from chumpy import Ch
import chumpy as ch
from chumpy.utils import row, col
from opendr.lighting import *
from opendr.util_tests import get_earthmesh, process
from opendr.camera import ProjectPoints
from collections import OrderedDict
from opendr.simple import *

visualize = False

def getcam():
    from opendr.camera import ProjectPoints

    w = 256
    h = 192

    f = np.array([200,200])
    rt = np.zeros(3)
    t = np.zeros(3)
    k = np.zeros(5)
    c = np.array([w/2., h/2.])

    if visualize == True:
        ratio = 640 / 256.
        f *= ratio
        c *= ratio
        w *= ratio
        h *= ratio

    pp = ProjectPoints(f=f, rt=rt, t=t, k=k, c=c)
    frustum = {'near': 0.01, 'far': 10.0, 'width': w, 'height': h}

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
if __name__=="__main__":

  fname_1='/Users/mudigonda/Data/faces/fwh/Tester_1/TrainingPose/pose_0.obj'
  fname_2='/Users/mudigonda/Data/faces/fwh/Tester_1/TrainingPose/pose_1.obj'
  fname_3='/Users/mudigonda/Data/faces/fwh/Tester_1/TrainingPose/pose_2.obj'
  from opendr.serialization import load_mesh
  verts_1, vt_1, faces = load_obj(fname_1)
  w,h = 200,200
  mesh_1 = load_mesh(fname_1)
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
  mesh_1.f = tris 
  mesh_1.vc = np.ones(np.float32(verts_1).shape)
  camera, frustum = getcam()
  light_pos = ch.array([0.,0.,1.2])
  A = LambertianPointLight(
            v=mesh_1.v,
            f=mesh_1.f,
            num_verts=len(mesh_1.v),
            light_pos=light_pos,
            vc=mesh_1.vc,
            light_color=np.array([1., 1., 1.]))
  bgcolor = np.array([0.,0.,0.])
  t_1 = ch.array([0.1,0.1,.9])
  f = ch.array([w/1.,h/1.])
  U = ProjectPoints(v=mesh_1.v,f=f,c=[w/2.,h/2.],k=ch.zeros(5),t=t_1,rt=ch.zeros(3))
  rn_1 = ColoredRenderer(vc=A,f=mesh_1.f, camera=U, frustum=frustum, bgcolor=bgcolor, num_channels=1)
  print('First Object rendered but now shown')
  plt.imshow(rn_1.r,cmap=cm.Greys_r,origin='lower')
  plt.show()

  #Face 2
  verts_2, vt_2, faces = load_obj(fname_2)

  #Face 3
  verts_3, vt_3, faces = load_obj(fname_3)
  mesh_3 = load_mesh(fname_3)
  mesh_3.f = tris 
  mesh_3.vc = np.ones(np.float32(verts_1).shape)

  verts = ch.zeros((11510,3,3))
  verts[:,:,0] = verts_1
  verts[:,:,1] = verts_2
  verts[:,:,2] = verts_3
  alpha = ch.zeros(3)
  alpha[0] = 0.5
  alpha[1] = 0.5
  alpha[2] = 0.5
  mesh_4 = mesh_3
  mesh_4.v = verts.dot(alpha)
  t = ch.array([0.,0.,.9])
  A = LambertianPointLight(
            v=mesh_3.v,
            f=mesh_3.f,
            num_verts=len(mesh_3.v),
            light_pos=light_pos,
            vc=mesh_3.vc,
            light_color=np.array([1., 1., 1.]))
  U = ProjectPoints(v=mesh_3.v,f=f,c=[w/2.,h/2.],k=ch.zeros(5),t=t,rt=ch.zeros(3))
  rn = ColoredRenderer(vc=A,f=mesh_3.f, camera=U, frustum=frustum, bgcolor=bgcolor, num_channels=1)
  print('New object rendered but now shown')
  #plt.imshow(rn.r,cmap=cm.Greys_r)
  trans, rot = ch.zeros(3), ch.array([1e-6,1e-6,1e-6])
  E_raw = rn - rn_1
  import IPython; IPython.embed()
  E_pyr = gaussian_pyramid(E_raw, n_levels = 5, normalization='SSE')
  free_variables = [alpha,t]
  idx = 0
  plt.clf()
  plt.imshow(rn.r, cmap=cm.Greys_r, origin='lower')
  plt.savefig('../tmp/pose_recover_CR_'+str(idx)+'.png')
  idx = idx + 1
  plt.clf()
  plt.imshow(rn_1.r, cmap=cm.Greys_r,origin='lower')
  plt.savefig('../tmp/pose_recover_CR_'+str(idx)+'.png')
  idx = idx + 1
  def cb(_):
    global idx
    global E_raw
    print ch.sum(E_raw**2)
    plt.clf()
    plt.imshow(np.abs(E_raw), cmap=cm.Greys_r,origin='lower')
    plt.savefig('../tmp/pose_recover_CR_'+str(idx)+'.png')
    idx = idx + 1
  #Setting objective for light
  #free_variables = [light_post,trans,rot]
  ch.minimize({'diff':E_pyr},x0=free_variables,callback = cb)
  print free_variables
  ch.minimize({'diff':E_raw},x0=free_variables,callback = cb)
  print free_variables
