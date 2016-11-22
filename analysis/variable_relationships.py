"""
This script is used just to look at relationships between the 
emulated variables. This is in an effort to reduce the dimensionality
of both the emulators and the covariance matrix.
"""
import numpy as np

#Names of variables
names = ["f0","f1","g0","g1"]

#Get the means and the variances
data_path = "../training_data/txt_files/"
data = np.loadtxt(data_path+"mean_models.txt")
var = np.loadtxt(data_path+"var_models.txt")
err = np.sqrt(var)
f0,f1,g0,g1 = data.T
f0_err,f1_err,g0_err,g1_err = err.T

import matplotlib.pyplot as plt
for i in range(len(names)):
    for j in xrange(i+1,len(names)):
        plt.errorbar(data[:,i],data[:,j],xerr=err[:,i],yerr=err[:,j],ls='',marker='o')
        plt.xlabel(names[i])
        plt.ylabel(names[j])
        plt.show()

