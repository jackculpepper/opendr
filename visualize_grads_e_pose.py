'''
Tries to visualize the gradients around a known solution space
useful to understand how gradients work
'''
import numpy as np
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import chumpy as ch
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
from PIL import Image
from opendr.lighting import LambertianPointLight
from opendr.renderer import ColoredRenderer
from opendr.camera import ProjectPoints
from opendr.simple import *

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

def show_grads(trans,input_im,verts,tris,vc):
#def show_grads(rotation,input_im,verts,tris,vc):
  #Set Camera
  w,h = 200,200
  camera = ProjectPoints(v=verts, rt=ch.array([0, 0, 0]), t=ch.array([0, 0, 2.]), f=ch.array([w,h])/2., c=ch.array([w,h])/2., k =ch.zeros(5))
  frustum = {'near': .01, 'far':10., 'width': w, 'height': h}
  #Set vertices
  light_post  = ch.array([1.,1.,1.])

  A  = LambertianPointLight(f=tris,
      v=verts,
      num_verts = len(verts),
      light_pos = light_post,
      vc = vc,
      light_color = ch.array([1.,1.,1.])
      )
  '''
  A = SphericalHarmonics(vn=VertNormals(v=verts,f=tris),
      components=[2.,1.,0.,0.,0.,0.,0.,0.,0.],
      light_color=ch.ones(3))
  '''
  rn = ColoredRenderer(vc=A, camera=camera,
      f=tris,bgcolor=[0.,0.,0.],frustum=frustum)
  rn.v +=  trans
  #rn.v = rn.v.dot(Rodrigues(rotation))
  diff_im = rn - input_im
  plt.imshow(np.abs(np.array(diff_im)))
  plt.draw()
  plt.show()
  sos = ch.sum(diff_im**2)
  sos_scale = sos/(diff_im.shape[0]*diff_im.shape[1]*diff_im.shape[2])
  E_pyr = gaussian_pyramid(diff_im,n_levels=6,normalization='SSE')
  sos_pyr = ch.sum(E_pyr**2)
  sos_pyr_scale = sos_pyr/(sos_pyr.shape[0])
  grads_pyr = sos_pyr_scale.dr_wrt(trans)
  #grads_pyr = sos_pyr_scale.dr_wrt(rotation)
  #grads_pyr = []
  # grads = sos_scale.dr_wrt(trans)
  grads = sos.dr_wrt(trans)
  #grads = sos_scale.dr_wrt(rotation)
  return grads_pyr,grads, rn.r, sos


if __name__=="__main__":
  #Modify this to point to where your data file is
  plt.ion()
  fname='/Users/mudigonda/Data/faces/fwh/Tester_1/TrainingPose/pose_0.obj'
  verts, vt, faces = load_obj(fname)
  verts = np.float32(verts)
  verts[:,2] -= 1.
  vt = np.float32(vt)
  vc = np.ones(verts.shape) #albedo
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
  #Read image file and display face
  #Modify this to where you place the pickled out from make_pose_data.py
  input_im = pickle.load(open('rendered_input.pkl','r'))
  #input_im = np.flipud(input_im)
  #Create Renderer
  delta = 1e-1
  start = 0. - delta
  stop = 0. + delta
  step = 1e-2
  x = np.arange(start,stop,step)
  y = np.arange(start,stop,step)
  u = np.zeros(x.shape)
  v = np.zeros(y.shape)
  e_u = np.zeros(x.shape)
  e_v = np.zeros(x.shape)
  idx1 =0
  for idx2 in np.arange(0,2):
    idx1 = 0
    print idx2
    for jj in np.arange(start,stop,step):
      if idx2 == 0:
        trans = ch.array([jj,0.,0.])
      else:
        trans = ch.array([0.,jj,0.])
      grads_pyr,grads, rendered_im, sos_scale = show_grads(trans,input_im,np.array(verts),tris,vc)
      if jj == 0:
          pickle.dump(rendered_im, open('rendered_input.pkl', 'w'), -1)
      print sos_scale
      if idx2 == 0:
        e_u[idx1] = sos_scale
        u[idx1] = grads[0][0]
      else:
        e_v[idx1] = sos_scale
        v[idx1] = grads[0][1]
      print grads, grads_pyr
      idx1+=1
    print idx1
  plt.plot(x,e_u,'r',label='Energy_x')
  plt.hold(True)
  plt.plot(x,e_v,'g',label='Energy_y')
  plt.legend()
  plt.savefig('tmp/energy_comparison_coarse.png')
  plt.clf()
  plt.plot(x,u,'r',label='grad_x')
  plt.hold(True)
  plt.plot(x,v,'g',label='grad_y')
  plt.legend()
  plt.savefig('tmp/grad_comparison_coarse.png')
  import IPython; IPython.embed()
