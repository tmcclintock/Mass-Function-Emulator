"""
A test script with which I can quickly switch between emulators.
"""
import emulator, sys, os
import numpy as np
sys.path.insert(0,'./visualization/')
import visualize
from mf_emulator import *

#Which data we are working with
dataname = "dfg"

#Read in the input cosmologies
cosmos = np.genfromtxt("./test_data/building_cosmos.txt")
#cosmos = np.delete(cosmos, [0, 5], 1) #Delete boxnum and ln10As
cosmos = np.delete(cosmos, 0, 1) #Delete boxnum
cosmos = np.delete(cosmos, -1, 0)#39 is broken
N_cosmos = len(cosmos)

#Read in the input data
database = "/home/tmcclintock/Desktop/Github_stuff/fit_mass_functions/output/%s/"%dataname
means     = np.loadtxt(database+"%s_means.txt"%dataname)
variances = np.loadtxt(database+"%s_vars.txt"%dataname)
data = np.ones((N_cosmos, len(means[0]),2)) #Last column is for mean/erros
data[:,:,0] = means
data[:,:,1] = np.sqrt(variances)

#Pick out the training/testing data
box, z_index = 0, 1
test_cosmo = cosmos[box]
test_data = data[box]
training_cosmos = np.delete(cosmos, box, 0)
training_data = np.delete(data, box, 0)

#Train
mf_emulator = mf_emulator("test")
mf_emulator.train(training_cosmos, training_data)

#Predict the TMF parameters
predicted = mf_emulator.predict_parameters(test_cosmo)
print "real params: ",test_data[:,0]
print "pred params: ",predicted[:,0]

#Read in the test mass function
MF_data = np.genfromtxt("./test_data/N_data/Box%03d_full/Box%03d_full_Z%d.txt"%(box, box, z_index))
lM_bins = MF_data[:,:2]
N_data = MF_data[:,2]
cov_data = np.genfromtxt("./test_data/covariances/Box%03d_cov/Box%03d_cov_Z%d.txt"%(box, box, z_index))
N_err = np.sqrt(np.diagonal(cov_data)) + 0.01*N_data

#Scale factors and redshifts
scale_factors = np.array([0.25,0.333333,0.5,0.540541,0.588235,0.645161,0.714286,0.8,0.909091,1.0])
redshifts = 1./scale_factors - 1.0

#Predict the TMF
volume = 1050.**3 #[Mpc/h]^3
n = mf_emulator.predict_mass_function(test_cosmo,redshift=redshifts[z_index],lM_bins=lM_bins, dataname=dataname)
N_emu = n*volume
        
chi2 = np.dot((N_data-N_emu),np.dot(np.linalg.inv(cov_data),(N_data-N_emu)))
sigdif = (N_data-N_emu)/N_err
for i in range(len(N_data)):
    print "Bin %d: %.1f +- %.1f\tvs\t%.1f  at  %f"%(i,N_data[i],N_err[i],N_emu[i],sigdif[i])
print "chi2 = %f"%chi2
    
lM = np.log10(np.mean(10**lM_bins,1))
visualize.N_comparison(lM, N_data, N_err, N_emu, show=True)


