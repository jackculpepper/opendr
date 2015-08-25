'''
This experiment  is to not employ chumpy minimize but use chumpy'gradients and SGD. It is useful for folks trying to do their own
optimization methods and fitting
'''
import numpy as np
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import chumpy as ch
import ipdb
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

def show_grads(trans,rotation,input_im,verts,tris,vc):
	#Set Camera
	w,h = 200,200 
	camera = ProjectPoints(v=verts, rt=ch.zeros(3), t=ch.array([0.,0.,1.]), f=ch.array([w,h])/2., c=ch.array([w,h])/2., k =ch.zeros(5))
	frustum = {'near': 0.01, 'far':10., 'width': w, 'height': h}
	#Set vertices
	light_post  = ch.zeros(3)
	
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
	rn.v = trans + rn.v.dot(Rodrigues(rotation))
	diff_im = rn - input_im
	sos = ch.sum(diff_im**2)
	sos_scale = sos/(diff_im.shape[0]*diff_im.shape[1]*diff_im.shape[2])
	E_pyr = gaussian_pyramid(diff_im,n_levels=5,normalization='SSE')
	sos_pyr = ch.sum(E_pyr**2)
	sos_pyr_scale = sos_pyr/(sos_pyr.shape[0])
	grads_pyr_trans = sos_pyr_scale.dr_wrt(trans)
	grads_trans = sos_scale.dr_wrt(trans)
	grads_pyr_rot = sos_pyr_scale.dr_wrt(rot)
	grads_rot = sos_scale.dr_wrt(rot)
	return grads_pyr_trans,grads_trans, grads_pyr_rot, grads_rot, rn.r



def fit_obj(trans,rot,grads_trans,grads_rot,idx):
	LR = 0.01
	if np.mod(idx,10) == 0:
		idx = idx/10.
	trans = trans + LR*grads_trans
	rot = rot + LR*grads_rot
	return trans

if __name__=="__main__":
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
	input_im = pickle.load(open('../tmp/rendered_input.pkl','r'))
	trans = ch.array([0.,0.,0.])
	rot = ch.array([1e-6,1e-6,1e-6])
	#Search for trans
	for ii in np.arange(100):
		print trans
		grads_pyr_trans,grads_trans, grads_pyr_rot, grads_rot, rendered_im = show_grads(trans,rot,input_im,np.array(verts),tris,vc)
		trans = fit_obj(trans,rot,grads_trans,grads_rot,ii)
	import IPython; IPython.embed()
