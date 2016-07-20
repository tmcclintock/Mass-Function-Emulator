print "Creating a jackknife covariance matrix using LOO emulators."
"""
Algorithm goes as follows:
Allocate a 4X4 covariance matrix.
Calculate (F - F_emu) where F is f0, f1, g0, g1
for each cosmology using its LOO emulator.
"""

import numpy as np
import sys
import matplotlib as m
import matplotlib.pyplot as plt
sys.path.insert(0,"../Emulator/")
import Emulator

Ncosmos = 39 #40th is broken
Nreds = 10

path_to_cosmos = "../cosmology_files/building_cosmos_no_z.txt"
cosmos = np.genfromtxt(path_to_cosmos).T
cosmos = np.delete(cosmos,39,1)
#Needs this shape to build the emulator from

#Name of emulators
f0_name = "f0_%03d_emu"
g0_name = "g0_%03d_emu"
f1_name = "f1_%03d_emu"
g1_name = "g1_%03d_emu"

#Full emulator paths
f0_full_name = "saved_emulators/f0_emu"
f1_full_name = "saved_emulators/f1_emu"
g0_full_name = "saved_emulators/g0_emu"
g1_full_name = "saved_emulators/g1_emu"
#Load the full emulators
space = np.ones_like(cosmos[:,0]) #A filler array
f0_full_emu = Emulator.Emulator(space,space,space)
f1_full_emu = Emulator.Emulator(space,space,space)
g0_full_emu = Emulator.Emulator(space,space,space)
g1_full_emu = Emulator.Emulator(space,space,space)
f0_full_emu.load(f0_full_name)
f1_full_emu.load(f1_full_name)
g0_full_emu.load(g0_full_name)
g1_full_emu.load(g1_full_name)

#Load in the data to be emulated
f0_means = np.genfromtxt("linear_fits/f0.txt")
g0_means = np.genfromtxt("linear_fits/g0.txt")
f1_means = np.genfromtxt("linear_fits/f1.txt")
g1_means = np.genfromtxt("linear_fits/g1.txt")
f0_vars = np.genfromtxt("linear_fits/f0_var.txt")
g0_vars = np.genfromtxt("linear_fits/g0_var.txt")
f1_vars = np.genfromtxt("linear_fits/f1_var.txt")
g1_vars = np.genfromtxt("linear_fits/g1_var.txt")
f0_err = np.sqrt(f0_vars)
f1_err = np.sqrt(f1_vars)
g0_err = np.sqrt(g0_vars)
g1_err = np.sqrt(g1_vars)

cov = np.zeros((4,4))
#Observed paramters f0, f1, g0, g1
obs = np.zeros((4,Ncosmos))
#Model aka emulated parameters
model = np.zeros((4,Ncosmos))

for i in xrange(0,Ncosmos):
    cosmo_predicted = cosmos[:,i]
    #f0_real = f0_means[i]
    #g0_real = g0_means[i]
    #f1_real = f1_means[i]
    #g1_real = g1_means[i]

    f0_real,f0_real_var = f0_full_emu.predict_one_point(cosmo_predicted)
    f1_real,f1_real_var = f1_full_emu.predict_one_point(cosmo_predicted)
    g0_real,g0_real_var = g0_full_emu.predict_one_point(cosmo_predicted)
    g1_real,g1_real_var = g1_full_emu.predict_one_point(cosmo_predicted)

    space = np.ones_like(cosmos[:,0]) #A filler array
    f0_emu = Emulator.Emulator(space,space,space)
    g0_emu = Emulator.Emulator(space,space,space)
    f1_emu = Emulator.Emulator(space,space,space)
    g1_emu = Emulator.Emulator(space,space,space)
    f0_emu.load("saved_emulators/leave_one_out_emulators/%s"%f0_name%(i))
    g0_emu.load("saved_emulators/leave_one_out_emulators/%s"%g0_name%(i))
    f1_emu.load("saved_emulators/leave_one_out_emulators/%s"%f1_name%(i))
    g1_emu.load("saved_emulators/leave_one_out_emulators/%s"%g1_name%(i))
    f0_test,f0_var = f0_emu.predict_one_point(cosmo_predicted)
    g0_test,g0_var = g0_emu.predict_one_point(cosmo_predicted)
    f1_test,f1_var = f1_emu.predict_one_point(cosmo_predicted)
    g1_test,g1_var = g1_emu.predict_one_point(cosmo_predicted)
    
    print "i = %d"%i
    print "f0real = %f +- %f   f0 = %f +- %f"%(f0_real,np.sqrt(f0_real_var),f0_test,np.sqrt(f0_var))
    print "g0real = %f +- %f   g0 = %f +- %f"%(g0_real,np.sqrt(g0_real_var),g0_test,np.sqrt(g0_var))
    print "f1real = %f +- %f   f1 = %f +- %f"%(f1_real,np.sqrt(f1_real_var),f1_test,np.sqrt(f1_var))
    print "g1real = %f +- %f   g1 = %f +- %f"%(g1_real,np.sqrt(g1_real_var),g1_test,np.sqrt(g1_var))
    print ""
    
    obs[0,i] = f0_real
    obs[1,i] = f1_real
    obs[2,i] = g0_real
    obs[3,i] = g1_real
    model[0,i] = f0_test
    model[1,i] = f1_test
    model[2,i] = g0_test
    model[3,i] = g1_test
diff = obs - model
pf = (Ncosmos-1.0)/Ncosmos #prefactor
for i in range(4):
    print diff[i]/model[i]
    for j in range(4):
        cov[i,j] = pf * np.sum(diff[i]*diff[j])
print cov
np.savetxt("../building_data/tinker_hyperparam_covariances.txt",cov)

inds = np.arange(Ncosmos)
plt.plot(inds,diff[0])
plt.plot(inds,diff[1])
plt.plot(inds,diff[2])
plt.plot(inds,diff[3])
plt.show()

corr = np.ones_like(cov)
for i in xrange(0,4):
    for j in xrange(0,4):
        corr[i,j] = cov[i,j]/np.sqrt(cov[i,i]*cov[j,j])
print corr

plt.pcolor(corr,vmin=-1.0,vmax = 1.0,cmap ='RdBu_r')
plt.show()
    
#Covariance goes f0, f1, g0, g1
def cov_fg(a,cov):
    k = 1./2.-a
    return cov[0,2]+k*(cov[0,3]+cov[1,2])+k*k*cov[1,3]
domain = np.arange(0,1,0.02)
plt.plot(domain,cov_fg(domain,cov))
plt.show()
