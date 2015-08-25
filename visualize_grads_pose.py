'''
useful to visualize quiver plots of how gradients work.

Lot of intuition on the complexity of fitting
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

def show_grads(trans,input_im,verts,tris,vc):
#def show_grads(rotation,input_im,verts,tris,vc):
	#Set Camera
	w,h = 200,200
	camera = ProjectPoints(v=verts, rt=ch.zeros(3), t=ch.zeros(3), f=ch.array([w,h])/2., c=ch.array([w,h])/2., k =ch.zeros(5))
	frustum = {'near': 0, 'far':10., 'width': w, 'height': h}
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
	sos = ch.sum(diff_im**2)
	sos_scale = sos/(diff_im.shape[0]*diff_im.shape[1]*diff_im.shape[2])
	E_pyr = gaussian_pyramid(diff_im,n_levels=6,normalization='SSE')
	sos_pyr = ch.sum(E_pyr**2)
	sos_pyr_scale = sos_pyr/(sos_pyr.shape[0])
	grads_pyr = sos_pyr_scale.dr_wrt(trans)
	#grads_pyr = sos_pyr_scale.dr_wrt(rotation)
	#grads_pyr = []
	grads = sos_scale.dr_wrt(trans)
	#grads = sos_scale.dr_wrt(rotation)
	return grads_pyr,grads, rn.r 


if __name__=="__main__":
	fname='/Users/mudigonda/Data/faces/fwh/Tester_1/TrainingPose/pose_0.obj'
	imname='/Users/mudigonda/Data/faces/fwh/Tester_1/TrainingPose/pose_0.png'
	rect_name = 'rect_0.txt'
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
	input_im = pickle.load(open('tmp/rendered_input.pkl','r'))
	#Create Renderer
	delta = 1e-3
	start = 0. - delta
	stop = 0. + delta
	step = 5e-4
	#x,y,z = np.meshgrid(np.arange(start,stop,step), np.arange(start,stop,step),np.arange(start,stop,step),indexing='ij')
	x,y = np.meshgrid(np.arange(start,stop,step), np.arange(start,stop,step),indexing='ij')
	u = np.zeros(x.shape)
	v = np.zeros(y.shape)
	#w = np.zeros(z.shape)
	u_pyr = np.zeros(x.shape)
	v_pyr = np.zeros(y.shape)
	idx1 =0 
	idx2 =0
	idx3 =0
	for ii in np.arange(start,stop,step):
		idx2=0
		for jj in np.arange(start,stop,step):
			#idx3=0
			#for kk in np.arange(start,stop,step):
				#print '=?',
				#print x[idx1,idx2], y[idx1,idx2]
				trans = ch.array([ii,0.,0.])
				#print '%.4f, %.4f, %.4f' % (trans[0], 0. , 0.)
				grads_pyr,grads, rendered_im = show_grads(trans,input_im,np.array(verts),tris,vc)
				print grads, grads_pyr
				#print '-> %.4f, %.4f, %.4f' % (grads[0][0], grads[0][1], grads[0][2]),
				#print '-> %.4f, %.4f, %.4f' % (grads_pyr[0][0], grads_pyr[0][1], grads_pyr[0][2])
				#u[idx1,idx2,idx3] = grads_pyr[0][0]
				#v[idx1,idx2,idx3] = grads_pyr[0][1]
				#w[idx1,idx2,idx3] = grads_pyr[0][2]
				#u[idx1,idx2] = grads[0][0]
				#v[idx1,idx2] = grads[0][1]
				print grads, grads_pyr
				diff_im = input_im - rendered_im
				diff_im -= diff_im.min()
				diff_im /= diff_im.max()
				'''
				plt.imshow(diff_im)
				plt.savefig('tmp/diff_im_'+str(idx1)+'_'+str(idx2)+'_'+str(idx3)+'.png')
				plt.clf()
				'''
				idx2 += 1
		idx1+=1
		print idx1
	idx = 0
	'''
	for ii in np.arange(start,stop,step):
		plt.quiver(x[:,:,ii],y[:,:,ii],u[:,:,ii],v[:,:,ii])
		plt.savefig('tmp/quiver_'+str(idx)+'.png')
		plt.axis('equal')
		idx +=1
		'''
	plt.quiver(x,y,u,v)
	plt.savefig('tmp/quiver.png')
	plt.axis('equal')
	import IPython; IPython.embed()
