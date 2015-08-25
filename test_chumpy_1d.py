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
for ii in np.arange(iter):
	df_dx_0 = val.dr_wrt(x[0])
	df_dx_1 = val.dr_wrt(x[1])
	df_dx_2 = val.dr_wrt(x[2])
	#x = x - ch.array(LR*df_dx.data.flatten())
	grad= np.array([LR*df_dx_0.data.flatten(),LR*df_dx_1.data.flatten(),LR*df_dx_2.data.flatten()])
	x.x = x.x - grad.flatten()
	print val
