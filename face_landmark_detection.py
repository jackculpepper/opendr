'''
This script will extract the landmarks for a given image
This is important if you want to work with real gray scale images otherwise experiments will be stuck in rendered image land
Mayur Mudigonda, Jack Culpepper
'''
import sys
import os
import dlib
import glob
import numpy as np
from PIL import Image
from scipy.io import savemat

predictor_path = 'shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

#f = 'pose_0.jpg'
#f = '/Users/mudigonda/Data/faces/fwh/Tester_1/TrainingPose/pose_0.png'
f = '/Users/mudigonda/Data/faces/fwh/Tester_5/TrainingPose/pose_1.png'
#f = '2009_004587.jpg'

print("Processing file: {}".format(f))
img = np.array(Image.open(f))

dets = detector(img, 1)
print("Number of faces detected: {}".format(len(dets)))

'''
size = 227
padding = 0.3

chips, shapes = predictor.get_chips_and_shapes(img,
                                               dets,
                                               size,
                                               padding)
'''
for jj,d in enumerate(dets):
	shape = predictor(img, d)
	#import IPython; IPython.embed()

	#with open('rect_%d.txt'%jj, 'w') as fh:
	with open('/Users/mudigonda/Projects/vision-shape/data/Tester_5_rect_%d.txt'%jj, 'w') as fh:
		fh.write('%d,'%shape.rect.left())
		fh.write('%d,'%shape.rect.top())
		fh.write('%d,'%shape.rect.right())
		fh.write('%d\n'%shape.rect.bottom())
	#with open('shape_%d.txt'%jj,'w') as fh:
	with open('/Users/mudigonda/Projects/vision-shape/data/Tester_5_shape_%d.txt'%jj,'w') as fh:
		for ii in range(shape.num_parts):
			fh.write('%d,%d,%d\n'%(ii,shape.part(ii).x, shape.part(ii).y))

