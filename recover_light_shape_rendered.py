'''
This works with the data gengerated in make_pose_data. Is a good unit test to have
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

if __name__=="__main__":
	fname='/Users/mudigonda/Data/faces/fwh/Tester_1/TrainingPose/pose_0.obj'
	rect_name = '../data/rect_0.txt'
	verts, vt, faces = load_obj(fname)
	verts = np.float32(verts)
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
	im = pickle.load(open('../tmp/rendered_input.pkl','r')) 

	#Now load the rectangle where we think the face is 
	for line in open(rect_name):
		vals = np.int32(line.split(','))
	
	band = 35 
	vals[0] = vals[0] - band
	vals[2] = vals[2] + band
	vals[1] = vals[1] - band
	vals[3] = vals[3] + band
	print vals
	#Crop out that part of the face
	w,h = 200,200 
	#Create Renderer
	#Set Camera
	rt , t = ch.zeros(3), ch.array([0.,0.,0.7])
	camera = ProjectPoints(v=verts, rt=rt, t=t, f=ch.array([w,h])/2., c=ch.array([w,h])/2., k =ch.zeros(5))
	#Set vertices
	trans, rot = ch.array([0.,0.,0.]), ch.array([0.,0.,-.2])
	frustum = {'near': 0.01, 'far':10., 'width': w, 'height': h}
	bgcolor = ch.zeros(3)
	#Set vertices
	light_post  = ch.array([0.,0.,2.])
	A = LambertianPointLight(f=tris,
			v=verts,
			num_verts = len(verts),
			light_pos = light_post,
			vc = vc,
			light_color = ch.array([1.,1.,1.])
			)
	rn = ColoredRenderer(vc=A,camera=camera,f=tris,bgcolor=bgcolor,frustum=frustum)
	idx = 0
	plt.imshow(rn.r,cmap=cm.Greys_r,origin='lower')
	plt.savefig('../tmp/pose_recover_CR_'+str(idx)+'.png')
	plt.clf()
	idx = idx+1
	plt.imshow(im,cmap=cm.Greys_r,origin='lower')
	plt.savefig('../tmp/pose_recover_CR_'+str(idx)+'.png')
	plt.clf()
	idx = idx+1
	rn.v = trans + rn.v.dot(Rodrigues(rot))
	import IPython; IPython.embed()
	def cb(_):
		global idx
		global E_raw
		print ch.sum(E_raw**2)
		plt.imshow(np.abs(E_raw), cmap=cm.Greys_r,origin='lower')
		plt.savefig('../tmp/pose_recover_CR_'+str(idx)+'.png')
		idx = idx + 1
	#Setting objective for light
	free_variables = [light_post]
	E_raw = ch.mean(im - rn,axis=2)
	E_pyr = gaussian_pyramid(E_raw, n_levels = 6, normalization='SSE')
	ch.minimize({'diff':E_pyr},x0=free_variables,callback = cb )
	print free_variables
	ch.minimize({'diff':E_raw},x0=free_variables,callback = cb)
	print free_variables
	#ch.minimize({'pyr':E_pyr},x0=free_variables,method='SLSQP',callback = cb, constraints=cons)
	print free_variables
	#ch.minimize({'diff':E_raw},x0=free_variables,method='SLSQP',callback = cb, constraints=cons)
	#print free_variables
	#Setting objective for pose
	free_variables = [trans,rot]
	ch.minimize({'diff':E_pyr},x0=free_variables,callback = cb )
	print free_variables
	ch.minimize({'diff':E_raw},x0=free_variables,callback = cb)
	print free_variables
	import IPython; IPython.embed()
	plt.imshow(rn.r,cmap=cm.Greys_r,origin='lower')
	plt.savefig('../tmp/pose_recover_CR_'+str(idx)+'.png')
