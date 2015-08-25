'''
This script will extract the landmarks for a given image
Mayur Mudigonda, Jack Culpepper
'''
import sys
import os
import dlib
import glob2
import numpy as np
from PIL import Image
import IPython
from scipy.io import savemat


if __name__ == "__main__":
	predictor_path = '../data/shape_predictor_68_face_landmarks.dat'

	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(predictor_path)
	output_path = '../data/'

	f = '/Users/mudigonda/Data/faces/fwh/Tester_5/TrainingPose/pose_1.png'
	files = glob2.glob('/Users/mudigonda/Data/faces/fwh/*/*/**png*')

	for aa in range(150):
		for bb in range(20):
			f = [instance_name for instance_name in files if '/Tester_'+str(aa+1)+'/TrainingPose/pose_'+str(bb) +'.png' in instance_name]
			print("Processing file: {}".format(f))
			img = np.array(Image.open(f[0],'r'))
			dets = detector(img, 1)
			print("Number of faces detected: {}".format(len(dets)))

			for jj,d in enumerate(dets):
				shape = predictor(img, d)
				#import IPython; IPython.embed()

				#with open('rect_%d.txt'%jj, 'w') as fh:
				with open('/Users/mudigonda/Projects/vision-shape/data/Tester_%d_pose_%d_rect_%d.txt'%(aa,bb,jj), 'w') as fh:
					fh.write('%d,'%shape.rect.left())
					fh.write('%d,'%shape.rect.top())
					fh.write('%d,'%shape.rect.right())
					fh.write('%d\n'%shape.rect.bottom())
				#with open('shape_%d.txt'%jj,'w') as fh:
				with open('/Users/mudigonda/Projects/vision-shape/data/Tester_%d_pose_%d_shape_%d.txt'%(aa,bb,jj),'w') as fh:
					for ii in range(shape.num_parts):
						fh.write('%d,%d,%d\n'%(ii,shape.part(ii).x, shape.part(ii).y))
