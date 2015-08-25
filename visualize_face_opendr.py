#Script to use all of OpenDR's demo to visualize faces
'''
this is a simple script to load and display a face
need to read a mesh, etc
'''

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

#Determines the type of visualization
FLAG = 0

if __name__=="__main__":

  fname='/Users/mudigonda/Data/faces/fwh/Tester_1/TrainingPose/pose_0.obj'
  from opendr.serialization import load_mesh
  verts, vt, faces = load_obj(fname)
  w,h = 200,200 
  mesh = load_mesh(fname)
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
  mesh.f = tris 
  mesh.vc = np.ones(np.float32(verts).shape)
  mesh.v = mesh.v *0.1
  mesh.v[:,2] += 0.4
  rot = ch.array([0.,0.,0.])
  trans = ch.array([0.,0.,0.])
  #mesh.v += mesh.v.dot(Rodrigues(rot))
  #light_post = ch.array([0.,0.,1.])
  bgcolor = np.array([0.,0.,0.])
  for jj in np.arange(-2,3,1):
    for ii in np.arange(-2,3,1):
      for kk in np.arange(-2,3,1):
          t = ch.array([0.,0.,ii])
          trans = ch.array([0.,0.,kk])
          #rt = ch.zeros(3)
          rt = ch.zeros(3)
          f = ch.array([w/2.,h/2.])
          light_post = ch.array([0.,0.,jj])
          A = LambertianPointLight(
                    v=mesh.v,
                    f=mesh.f,
                    num_verts=len(mesh.v),
                    light_pos=light_post,
                    vc=mesh.vc,
                    light_color=np.array([1.,1.,1. ]))
          frustum = {'near':0.01,'far':10.0,'width':w,'height':h}
          #frustum = {'near':-3.0,'far':3.0,'width':w,'height':h}
          U = ProjectPoints(v=mesh.v,f=f,c=[w/2.,h/2.],k=ch.zeros(5),t=t,rt=rt)
          rn = ColoredRenderer(vc=A,f=mesh.f, camera=U, frustum=frustum, bgcolor=bgcolor, num_channels=1)
          rn.v = trans + rn.v.dot(Rodrigues(rot))
          #rn.v += push
          plt.imshow(rn.r,cmap=cm.Greys_r,origin='lower')
          plt.savefig('../tmp/scale_face_'+str(ii)+'_'+str(jj)+'_'+str(kk)+'.png')
  '''
  for ii in np.arange(10,150,10):
    f = ch.array([ii,ii])
    t = ch.array([0.,0.,1.])
    rt = ch.zeros(3)
    A = LambertianPointLight(
              v=mesh.v,
              f=mesh.f,
              num_verts=len(mesh.v),
              light_pos=light_post,
              vc=mesh.vc,
              light_color=np.array([1., 1., 1.]))
    frustum = {'near':0.01,'far':20.0,'width':w,'height':h}
    U = ProjectPoints(v=mesh.v,f=f,c=[w/2.,h/2.],k=ch.zeros(5),t=t,rt=rt)
    rn = ColoredRenderer(vc=A,f=mesh.f, camera=U, frustum=frustum, bgcolor=bgcolor, num_channels=1)
    plt.imshow(rn.r,cmap=cm.Greys_r,origin='lower')
    plt.savefig('../tmp/face_'+str(ii)+'_'+'.png')
    '''
