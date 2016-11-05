"""
This takes the training f(z) and g(z) points and
creates linear fits according to f(a) = f_0 + (0.5-a)*f_1
and g(a) g_0 + (0.5-a)*g_1.
"""
import numpy as np
import matplotlib.pyplot as plt

visualize = False

#These are the scale factors of the snapshots
scale_factors = np.array([0.25,0.333333,0.5,0.540541,0.588235,0.645161,0.714286,0.8,0.909091,1.0])
redshifts = 1./scale_factors - 1.0

#Paths to the linear data
training_data_path = "txt_files/mean_models.txt"
training_vars_path = "txt_files/var_models.txt"

#Get the data
data = np.loadtxt(training_data_path)
var = np.loadtxt(training_vars_path)

#The number of fits and how many points
N_boxes, N_z = 39, 10 #39th is broken

#Specify the pivot
ap = 0.5

#Create the output array
names = [r"$f$",r"$g$"]
ndim = 2
from_scratch = False
if from_scratch:
    lines = np.zeros((N_boxes,4)) #f0,f1,g0,g1
    covs = np.zeros((N_boxes,2,2,2)) #2 params: p0p0,p0p1,p1p0,p1p1
    np.savetxt("txt_files/lines.txt",lines,header="f0\tf1\tg0\tg1")
    np.save("txt_files/line_covs",covs)
else:
    lines = np.loadtxt("txt_files/lines.txt")
    covs = np.load("txt_files/line_covs.npy")

#Loop over cosmologies
box_lo,box_hi = 0,N_boxes
for i in range(box_lo,box_hi):
    for j in range(0,ndim):
        p = data[i*N_z:(i+1)*N_z,j]
        p_var = var[i*N_z:(i+1)*N_z,j]
        
        ones = np.ones((N_z))
        A = np.vstack((ones,(scale_factors-ap)))
        C = np.diag(p_var)
        Cinv = np.linalg.inv(C)
        left = np.linalg.inv(np.dot(A,np.dot(Cinv,A.T)))
        right = np.dot(A,np.dot(Cinv,p))
        b,m = np.dot(left,right)
        lines[i,j*2:j*2+2] = [b,m]
        covs[i,j] = np.linalg.inv(np.dot(A,np.dot(Cinv,A.T)))
        print "i=%d\tb,m = %s"%(i,lines[i,j*2:j*2+2])
        print covs[i,j]

        if visualize:
            x = np.linspace(min(scale_factors),max(scale_factors),50)
            model = b+(x-ap)*m
            plt.errorbar(scale_factors,p,np.sqrt(p_var))
            plt.plot(x,model)
            plt.ylabel(names[j])
            plt.xlabel(r"$a$")
            plt.title(r"Box%03d for %s"%(i,names[j]))
            plt.show()

#Save everything
np.savetxt("txt_files/lines.txt",lines,header="f0\tf1\tg0\tg1")
np.save("txt_files/line_covs",covs)
