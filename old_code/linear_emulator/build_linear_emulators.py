"""
This is a short script that creates full emulators for
f0, f1, g0 and g1 and then predicts a cosmology.
"""

import numpy as np
import sys
import emulator as Emulator

scale_factors = np.array([0.25,0.333333,0.5,0.540541,0.588235,0.645161,0.714286,0.8,0.909091,1.0])
redshifts = 1./scale_factors - 1.0
volume = 1.e9 #(1000.)**3 #(Mpc/h)^3

cosmo_path = "../cosmology_files/building_cosmos_no_z.txt"
cosmos = np.genfromtxt(cosmo_path)
cosmos = np.delete(cosmos,39,0)
#Need to remove last cosmology because it broke

param_names = [r"$\Omega_bh^2$",r"$\Omega_ch^2$",r"$w_0$",r"$n_s$",r"$\log_{10}A_s$",r"$H_0$",r"$N_{eff}$",r"$\sigma_8$"]
simple_param_names = ["Omegabh2","Omegach2","w0","ns","log10As","H0","Neff","sigma8"]

build_emulators = True

f0_name = "f0_emu"
g0_name = "g0_emu"
f1_name = "f1_emu"
g1_name = "g1_emu"

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

if build_emulators:
    print "Training Emulators"
    print cosmos.shape, f0_means.shape
    f0_emu = Emulator.Emulator(name=f0_name,xdata=cosmos,ydata=f0_means,yerr=f0_err)
    f1_emu = Emulator.Emulator(name=f1_name,xdata=cosmos,ydata=f1_means,yerr=f1_err)
    g0_emu = Emulator.Emulator(name=g0_name,xdata=cosmos,ydata=g0_means,yerr=g0_err)
    g1_emu = Emulator.Emulator(name=g1_name,xdata=cosmos,ydata=g1_means,yerr=g1_err)
    
    print "\tTraining %s"%f0_name
    f0_emu.train()
    print "\tSaving %s"%f0_name
    f0_emu.save("saved_emulators/%s"%f0_name)
    print "\t%s saved"%f0_name
    print "\tTraining %s"%f1_name
    f1_emu.train()
    print "\tSaving %s"%f1_name
    f1_emu.save("saved_emulators/%s"%f1_name)
    print "\t%s saved"%f1_name
    print "\tTraining %s"%g0_name
    g0_emu.train()
    print "\tSaving %s"%g0_name
    g0_emu.save("saved_emulators/%s"%g0_name)
    print "\t%s saved"%g0_name
    print "\tTraining %s"%g1_name
    g1_emu.train()
    print "\tSaving %s"%g1_name
    g1_emu.save("saved_emulators/%s"%g1_name)
    print "\t%s saved"%g1_name
