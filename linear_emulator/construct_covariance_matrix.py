print "Creating a jackknife covariance matrix using LOO emulators."
"""
Algorithm goes as follows:
Allocate a 4X4 covariance matrix.
Calculate (F - F_emu) where F is f0, f1, g0, g1
for each cosmology using its LOO emulator.
"""

showplots = True

import numpy as np
import sys
import matplotlib as m
import matplotlib.pyplot as plt
#plt.rc('text',usetex=True, fontsize=20)
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

#Observed paramters f0, f1, g0, g1
obs = np.zeros((4,Ncosmos))
#Model aka emulated parameters
model = np.zeros((4,Ncosmos))

#First loop over each cosmology and add its LOO emulator to a list
f0_list = []
f1_list = []
g0_list = []
g1_list = []
for i in xrange(0,Ncosmos):
    space = np.ones_like(cosmos[:,0]) #A filler array
    f0_list.append(Emulator.Emulator(space,space,space))
    f1_list.append(Emulator.Emulator(space,space,space))
    g0_list.append(Emulator.Emulator(space,space,space))
    g1_list.append(Emulator.Emulator(space,space,space))
    f0_list[i].load("saved_emulators/leave_one_out_emulators/%s"%f0_name%(i))
    g0_list[i].load("saved_emulators/leave_one_out_emulators/%s"%g0_name%(i))
    f1_list[i].load("saved_emulators/leave_one_out_emulators/%s"%f1_name%(i))
    g1_list[i].load("saved_emulators/leave_one_out_emulators/%s"%g1_name%(i))
    
for i in xrange(0,Ncosmos):
    cosmo_predicted = cosmos[:,i]
    f0_test,f0_var = f0_list[i].predict_one_point(cosmo_predicted)
    g0_test,g0_var = g0_list[i].predict_one_point(cosmo_predicted)
    f1_test,f1_var = f1_list[i].predict_one_point(cosmo_predicted)
    g1_test,g1_var = g1_list[i].predict_one_point(cosmo_predicted)
    model[0,i] = f0_test
    model[1,i] = f1_test
    model[2,i] = g0_test
    model[3,i] = g1_test

    #obs[0,i] = f0_means[i]
    #obs[1,i] = f1_means[i]
    #obs[2,i] = g0_means[i]
    #obs[3,i] = g1_means[i]
    #f0_full,f0_full_var = f0_full_emu.predict_one_point(cosmo_predicted)
    #f1_full,f1_full_var = f1_full_emu.predict_one_point(cosmo_predicted)
    #g0_full,g0_full_var = g0_full_emu.predict_one_point(cosmo_predicted)
    #g1_full,g1_full_var = g1_full_emu.predict_one_point(cosmo_predicted)
    f0_mean = f1_mean = g0_mean = g1_mean = 0
    for j in xrange(0,Ncosmos):
        f0_test,f0_var = f0_list[j].predict_one_point(cosmo_predicted)
        g0_test,g0_var = g0_list[j].predict_one_point(cosmo_predicted)
        f1_test,f1_var = f1_list[j].predict_one_point(cosmo_predicted)
        g1_test,g1_var = g1_list[j].predict_one_point(cosmo_predicted)
        f0_mean += f0_test/Ncosmos
        f1_mean += f1_test/Ncosmos
        g0_mean += g0_test/Ncosmos
        g1_mean += g1_test/Ncosmos
        continue #end j
    obs[0,i] = f0_mean
    obs[1,i] = f1_mean
    obs[2,i] = g0_mean
    obs[3,i] = g1_mean
    continue #end i

indices = np.arange(Ncosmos)
diff = obs - model
names = ["f0","f1","g0","g1"]
for i in range(4):
    plt.plot(indices,diff[i]/model[i],marker='o',ls='',label=names[i])
plt.legend()
plt.xlabel("Simulation number")
plt.ylabel("% diff")
if showplots:
    plt.show()
plt.close()

pf = (Ncosmos-1.0)/Ncosmos #prefactor
cov = np.zeros((4,4))
print diff.shape
for i in range(4):
    for j in range(4):
        cov[i,j] = pf * np.sum(diff[i]*diff[j])/Ncosmos
print cov
np.savetxt("../building_data/tinker_hyperparam_covariances.txt",cov)

corr = np.ones_like(cov)
for i in xrange(0,4):
    for j in xrange(0,4):
        corr[i,j] = cov[i,j]/np.sqrt(cov[i,i]*cov[j,j])
#print corr

plt.pcolor(corr,vmin=-1.0,vmax = 1.0,cmap ='RdBu_r')
if showplots:
    plt.show()
plt.close()    

#Covariance goes f0, f1, g0, g1
def cov_fg(a,cov):
    k = 1./2.-a
    return cov[0,2]+k*(cov[0,3]+cov[1,2])+k*k*cov[1,3]
domain = np.arange(0,1,0.02)
plt.plot(domain,cov_fg(domain,cov))
plt.ylabel(r"$C(f,g)$")
plt.xlabel(r"$a$")
plt.subplots_adjust(left=0.2,bottom=0.15)
if showplots:
    plt.show()
plt.close()
