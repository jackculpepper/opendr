'''
This is to render a face and store it as a stand alone example to do fitting. It is a helper script
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
from PIL import Image
from opendr.lighting import LambertianPointLight 
from opendr.renderer import ColoredRenderer
from opendr.camera import ProjectPoints
from opendr.simple import *
import pickle

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
	imname='/Users/mudigonda/Data/faces/fwh/Tester_1/TrainingPose/pose_0.png'
	rect_name = 'rect_0.txt'
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
	w,h = 200,200 
	#Create Renderer
	#Set Camera
	#Set vertices
	frustum = {'near': 0.01, 'far':10., 'width': w, 'height': h}
	#trans_fix, rot_fix = ch.array([0.1,0.,0.0]), ch.array([0.002,0.002,0.002])
	trans_fix = ch.array([0.,0.,0.])
	rotation = ch.array([0.,0.,-0.5])
	#camera = ProjectPoints(v=verts, rt=ch.zeros(3), t=ch.zeros(3), f=ch.array([w,h])/2., c=ch.array([w,h])/2., k =ch.zeros(5))
	camera = ProjectPoints(v=verts, rt=ch.zeros(3), t=ch.array([0.,0.,.7]), f=ch.array([w,h])/2., c=ch.array([w,h])/2., k =ch.zeros(5))
	#rn.v = verts
	#Set vertices
	light_post  = ch.array([0.,0.,1.])
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
	rn = ColoredRenderer(vc=A,camera=camera,
			f=tris,bgcolor=[0.,0.,0.],frustum = frustum)
	#rn.v = rn.v + trans_fix 
	#rn.v += rn.v.dot(Rodrigues(rotation)) 
	rn.v = trans_fix + rn.v.dot(Rodrigues(rotation))
	input_im = rn.r
	pickle.dump(input_im,open('../tmp/rendered_input.pkl','w'))
	plt.imshow(rn.r,cmap=cm.Greys_r,origin='lower')
	plt.savefig('../tmp/pickled_input_im.png')
