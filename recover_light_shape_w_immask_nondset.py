'''
This was the first experiment using the facelandmarkdetection pipeline on novel images
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
	fname='/Users/mudigonda/Data/faces/fwh/Tester_2/TrainingPose/pose_0.obj'
	imname='../tmp/gerry.jpg'
	rect_name = '../tmp/gerry_rect_0.txt'
	shape_name = '../tmp/gerry_shape_0.txt'
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
	im_gray = cv2.imread(imname,cv2.CV_LOAD_IMAGE_GRAYSCALE) 
	
	vals_list = []
	for line in open(shape_name,'r'):
		vals_list.append(line.split(','))
	
	vals = np.asarray(vals_list,dtype='int32')
	hull = cv2.convexHull(np.float32(vals[:,1:3]))

	im = np.zeros(im_gray.shape)
	for ii in np.arange(im.shape[0]):
		for jj in np.arange(im.shape[1]):
			flag=  cv2.pointPolygonTest(hull,(jj,ii),True)
			if flag >= 0:
				im[ii,jj]=im_gray[ii,jj]
	#Now load the rectangle where we think the face is 
	for line in open(rect_name):
		vals = np.int32(line.split(','))
	
	band = 0 
	vals[0] = vals[0] - band
	vals[2] = vals[2] + band
	vals[1] = vals[1] - band
	vals[3] = vals[3] + band
	print vals
	#Crop out that part of the face
	im = im[vals[1]:vals[3], vals[0]:vals[2]]
	im = im/np.float(im.max())
	w,h = im.shape
	M = cv2.getRotationMatrix2D((w/2,h/2), 180, 1.0)
	im_rot = cv2.warpAffine(im, M, (w,h))
	#Create Renderer
	#Set Camera
	rt , t = ch.zeros(3), ch.array([0.,0.,.6])
	f = ch.array([w/2.,h/2.])
	camera = ProjectPoints(v=verts, rt=rt, t=t, f=f, c=ch.array([w,h])/2., k =ch.zeros(5))
	#Set vertices
	trans, rot = ch.array([0.,0.,0.]), ch.array([0.,0.,0.])
	frustum = {'near': 0.01, 'far':10., 'width': w, 'height': h}
	bgcolor = ch.zeros(3)
	#Set vertices
	light_post  = ch.array([0.,0.,1.7])
	A = LambertianPointLight(f=tris,
			v=verts,
			num_verts = len(verts),
			light_pos = light_post,
			vc = vc,
			light_color = ch.array([1.,1.,1.])
			)
	rn = ColoredRenderer(vc=A,camera=camera,f=tris,bgcolor=bgcolor,frustum=frustum)
	idx = 0
	rn.v = trans + rn.v.dot(Rodrigues(rot))
	import IPython; IPython.embed()
	plt.clf()
	plt.imshow(rn.r, cmap=cm.Greys_r, origin='lower')
	plt.savefig('../tmp/pose_recover_CR_'+str(idx)+'.png')
	idx = idx + 1
	plt.clf()
	plt.imshow(im_rot, cmap=cm.Greys_r,origin='lower')
	plt.savefig('../tmp/pose_recover_CR_'+str(idx)+'.png')
	idx = idx + 1
	import IPython; IPython.embed()
	def cb(_):
		global idx
		global E_raw
		print ch.sum(E_raw**2)
		plt.clf()
		plt.imshow(np.abs(E_raw), cmap=cm.Greys_r,origin='lower')
		plt.savefig('../tmp/pose_recover_CR_'+str(idx)+'.png')
		idx = idx + 1
	#Setting objective for light
	free_variables = [light_post]
	E_raw = im_rot - ch.mean(rn,axis=2) 
	E_pyr = gaussian_pyramid(E_raw, n_levels = 5, normalization='SSE')
	#E_pyr = laplacian_pyramid(E_raw, n_levels = 6, imshape=im_rot.shape, as_list=False, normalization=None)
	#ch.minimize({'pyr':E_pyr},x0=free_variables,callback = cb)
	print free_variables
	#ch.minimize({'diff':E_raw},x0=free_variables,method='SLSQP',callback = cb, constraints=cons)
	#print free_variables
	#Setting objective for pose
	free_variables = [light_post,trans,rot,f]
	#free_variables = [light_post,trans,rot]
	ch.minimize({'diff':E_pyr},x0=free_variables,callback = cb)
	print free_variables
	ch.minimize({'diff':E_raw},x0=free_variables,callback = cb)
	print free_variables
	import IPython; IPython.embed()
	plt.clf()
	plt.imshow(rn.r,cmap=cm.Greys_r,origin='lower')
	plt.savefig('../tmp/pose_recover_CR_'+str(idx)+'.png')
