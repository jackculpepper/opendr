#SCript to test chumpy out
#Import Statements
import chumpy as ch
import numpy as np
import IPython 

x,y,A = ch.array([10,20,30]), ch.array([5]), ch.eye(3)

z = x.T.dot(A).dot(x)
val = z + y**2

iter = 1000
LR = 0.01
IPython.embed()
for ii in np.arange(iter):
	df_dx = val.dr_wrt(x)
	print df_dx.data.flatten()
	IPython.embed()
	#x = x - ch.array(LR*df_dx.data.flatten())
	x.x = x.x - LR*df_dx.data.flatten() 
	print val
	print x
