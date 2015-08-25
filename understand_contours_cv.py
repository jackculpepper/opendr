'''
Simple script to use landmark detection to crop a face
'''
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages/')
import cv2
import numpy as np
import IPython
import matplotlib.pyplot as plt
import matplotlib.cm as cm


if __name__=="__main__":
	rect_name = '../data/rect_0.txt'
	shape_name = '../data/shape_0.txt'
	imname='/Users/mudigonda/Data/faces/fwh/Tester_1/TrainingPose/pose_0.png'
	im = cv2.imread(imname,cv2.CV_LOAD_IMAGE_GRAYSCALE)
	#cv2.imshow("im",im)
	#cv2.waitKey(0)
	vals_list = []
	for line in open(shape_name,'r'):
		vals_list.append(line.split(','))
	
	vals = np.asarray(vals_list,dtype='int32')
	hull = cv2.convexHull(np.float32(vals[:,1:3]))

	masked_im = np.ones(im.shape)*255.
	IPython.embed()
	for ii in np.arange(masked_im.shape[0]):
		for jj in np.arange(masked_im.shape[1]):
			flag=  cv2.pointPolygonTest(hull,(jj,ii),True)
			if flag >= 5:
				masked_im[ii,jj]=im[ii,jj]


	plt.imshow(masked_im,cmap=cm.Greys_r)
	for ii in np.arange(len(hull)):
		plt.plot(vals[ii,1],vals[ii,2],'r+')
		plt.annotate(str(ii),xy=(vals[ii,1],vals[ii,2]),xytext=(vals[ii,1]+5,vals[ii,2]+5))

	plt.savefig('../tmp/cropped_im.png')
