'''
This is a cute visualization script that moves the light to show the plethora faces that can be produced with a change in light
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
	#Read image file and display face
	im = cv2.imread(imname,cv2.CV_LOAD_IMAGE_GRAYSCALE) 

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
	im = im[vals[1]:vals[3], vals[0]:vals[2]]
	im = im/np.float(im.max())
	im = im - 1.0
	w,h = im.shape
	#Display face
	#plt.imshow(im,cmap=cm.Greys_r)
	#plt.show()
	#Create Renderer
	rn = ColoredRenderer()
	#Set Camera
	rn.camera = ProjectPoints(v=verts, rt=ch.zeros(3), t=ch.zeros(3), f=ch.array([w,h])/2., c=ch.array([w,h])/2., k =ch.zeros(5))
	#Set vertices
	print "This is the minimum value we are adding (subtracting) so it steps away from the light"
	print verts[:,2].min()
	#verts[:,2] += verts[:,2].min()
	verts[:,2] -= 1.
	#verts[:,1] *= -1.
	rn.frustum = {'near': 0, 'far':10., 'width': w, 'height': h}
	rn.v = verts
	rn.f = tris
	rn.bgcolor = ch.zeros(3)
	#Set vertices
	light_post  = ch.ones(3)
	rn.vc = LambertianPointLight(f=tris,
			v=verts,
			num_verts = len(verts),
			light_pos = light_post,
			vc = vc,
			light_color = ch.array([1.,1.,1.])
			)


	f, ax = plt.subplots(1,3,sharey=True)
	f.hold(True)
	ax[0].imshow(rn.r,cmap=cm.Greys_r)
	ax[0].set_title('Light Pos at 1,1,1')
	import IPython; IPython.embed()

	light_post  = ch.array([0,0,0])
	rn.vc = LambertianPointLight(f=tris,
			v=verts,
			num_verts = len(verts),
			light_pos = light_post,
			vc = vc,
			light_color = ch.array([1.,1.,1.])
			)

	ax[1].imshow(rn.r,cmap=cm.Greys_r)
	ax[1].set_title('Light Pos at 0,0,0')

	light_post  = ch.array([-1,-1,-1])
	rn.vc = LambertianPointLight(f=tris,
			v=verts,
			num_verts = len(verts),
			light_pos = light_post,
			vc = vc,
			light_color = ch.array([1.,1.,1.])
			)

	ax[2].imshow(rn.r,cmap=cm.Greys_r)
	ax[2].set_title('Light Pos at -1, -1, - 1')

	plt.show()
