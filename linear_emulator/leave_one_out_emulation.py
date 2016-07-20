"""
This builds the leave-one-out emulators.
"""
import numpy as np
import sys
import matplotlib as m
import matplotlib.pyplot as plt
sys.path.insert(0,"../NM_model/")
sys.path.insert(0,"../Emulator/")
sys.path.insert(0,"../visualization/")
import Emulator
import NM_model as NM_model_module
import visualize

Ncosmos = 39 #40th is broken
Nreds = 10

scale_factors = np.array([0.25,0.333333,0.5,0.540541,0.588235,0.645161,0.714286,0.8,0.909091,1.0])
redshifts = 1./scale_factors - 1.0
volume = 1.e9 #(1000.)**3 #(Mpc/h)^3

data_base = "/home/tmcclintock/Desktop/Mass_Function_Data/"
data_path = data_base+"full_mf_data/Box%03d_full/Box%03d_full_Z%d.txt"
cov_path = data_base+"/covariances/Box%03d_cov/Box%03d_cov_Z%d.txt"

build_emulators = False
examine_emulators = True

path_to_cosmos = "../cosmology_files/building_cosmos_no_z.txt"
cosmos = np.genfromtxt(path_to_cosmos).T
cosmos = np.delete(cosmos,39,1)
#Needs this shape to build the emulator from

#This is the scale factor pivot
pivot = 0.5

#Name of emulators
f0_name = "f0_%03d_emu"
g0_name = "g0_%03d_emu"
f1_name = "f1_%03d_emu"
g1_name = "g1_%03d_emu"

#Load in the data to be emulated
f0_means = np.genfromtxt("linear_fits/f0.txt")
g0_means = np.genfromtxt("linear_fits/g0.txt")
f1_means = np.genfromtxt("linear_fits/f1.txt")
g1_means = np.genfromtxt("linear_fits/g1.txt")
f0_vars = np.genfromtxt("linear_fits/f0_var.txt")
g0_vars = np.genfromtxt("linear_fits/g0_var.txt")
f1_vars = np.genfromtxt("linear_fits/f1_var.txt")
g1_vars = np.genfromtxt("linear_fits/g1_var.txt")
f0f1_covars = np.genfromtxt("linear_fits/f0f1_covar.txt")
g0g1_covars = np.genfromtxt("linear_fits/g0g1_covar.txt")
f0_err = np.sqrt(f0_vars)
f1_err = np.sqrt(f1_vars)
g0_err = np.sqrt(g0_vars)
g1_err = np.sqrt(g1_vars)


"""
Loop over each cosmology and
build an emulator without that cosmology included.

Then save the emulator.
"""

for i in xrange(0,Ncosmos):
    if build_emulators:
        cosmos_used = np.delete(cosmos,i,1)
        f0_means_used = np.delete(f0_means,i)
        g0_means_used = np.delete(g0_means,i)
        f1_means_used = np.delete(f1_means,i)
        g1_means_used = np.delete(g1_means,i)
        f0_vars_used = np.delete(f0_vars,i)
        g0_vars_used = np.delete(g0_vars,i)
        f1_vars_used = np.delete(f1_vars,i)
        g1_vars_used = np.delete(g1_vars,i)
        f0_err_used = np.delete(f0_err,i)
        g0_err_used = np.delete(g0_err,i)
        f1_err_used = np.delete(f1_err,i)
        g1_err_used = np.delete(g1_err,i)
        
        f0_emu = Emulator.Emulator(name=f0_name%(i),xdata=cosmos_used,ydata=f0_means_used,yerr=f0_err_used)
        f1_emu = Emulator.Emulator(name=f1_name%(i),xdata=cosmos_used,ydata=f1_means_used,yerr=f1_err_used)
        g0_emu = Emulator.Emulator(name=g0_name%(i),xdata=cosmos_used,ydata=g0_means_used,yerr=g0_err_used)
        g1_emu = Emulator.Emulator(name=g1_name%(i),xdata=cosmos_used,ydata=g1_means_used,yerr=g1_err_used)
        
        print "\tTraining %s"%f0_name%(i)
        f0_emu.train()
        print "\tSaving %s"%f0_name%(i)
        f0_emu.save("saved_emulators/leave_one_out_emulators/%s"%f0_name%(i))
        print "\t%s saved"%f0_name%(i)
        print "\tTraining %s"%f1_name%(i)
        f1_emu.train()
        print "\tSaving %s"%f1_name%(i)
        f1_emu.save("saved_emulators/leave_one_out_emulators/%s"%f1_name%(i))
        print "\t%s saved"%f1_name%(i)
        print "\tTraining %s"%g0_name%(i)
        g0_emu.train()
        print "\tSaving %s"%g0_name%(i)
        g0_emu.save("saved_emulators/leave_one_out_emulators/%s"%g0_name%(i))
        print "\t%s saved"%g0_name%(i)
        print "\tTraining %s"%g1_name%(i)
        g1_emu.train()
        print "\tSaving %s"%g1_name%(i)
        g1_emu.save("saved_emulators/leave_one_out_emulators/%s"%g1_name%(i))
        print "\t%s saved"%g1_name%(i)

    if examine_emulators:
        print "Examining %d"%i
        cosmo_predicted = cosmos[:,i]
        f0_real = f0_means[i]
        g0_real = g0_means[i]
        f1_real = f1_means[i]
        g1_real = g1_means[i]
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
        #print "f0real = %f   f0 = %f +- %f"%(f0_real,f0_test,np.sqrt(f0_var))
        #print "g0real = %f   g0 = %f +- %f"%(g0_real,g0_test,np.sqrt(g0_var))
        #print "f1real = %f   f1 = %f +- %f"%(f1_real,f1_test,np.sqrt(f1_var))
        #print "g1real = %f   g1 = %f +- %f"%(g1_real,g1_test,np.sqrt(g1_var))

        #Build a cosmo_dict
        h = cosmo_predicted[5]/100.
        Om = (cosmo_predicted[0]+cosmo_predicted[1])/h**2
        Ob = cosmo_predicted[0]/h**2
        sigma8 = cosmo_predicted[7]
        w0 = cosmo_predicted[2]
        ns = cosmo_predicted[3]
        cosmo_dict = {"om":Om,"ob":Ob,"ol":1-Om,\
                          "ok":0.0,"h":h,"s8":sigma8,\
                          "ns":ns,"w0":w0,"wa":0.0}

        #Read in the hyperparameter covariances
        param_cov = np.loadtxt("../building_data/tinker_hyperparam_covariances.txt")

        for z_index in xrange(0,Nreds):
            print "\tExamining at Z%d"%z_index
            redshift = redshifts[z_index]
            sf = 1./(1.+redshift)
            f_test = f0_test + (pivot-sf)*f1_test
            f_var = f0_var + (pivot-sf)**2*f1_var 
            g_test = g0_test + (pivot-sf)*g1_test
            g_var = g0_var + (pivot-sf)**2*g1_var

            #Read in the NM data
            indata = np.loadtxt(data_path%(i,i,z_index))
            cov = np.loadtxt(cov_path%(i,i,z_index))
            lM_low,lM_high = indata[:,0],indata[:,1]
            lM_bins = np.array([lM_low,lM_high]).T
            NM_data = indata[:,2]
            
            #Remove bad data
            good_indices = np.where(NM_data > 0)[0]
            lM_bins = lM_bins[good_indices]
            bounds = np.array([np.min(lM_bins),np.max(lM_bins)])
            NM_data = NM_data[good_indices]
            cov = cov[:,good_indices]
            cov = cov[good_indices,:]
            NM_err = np.sqrt(np.diagonal(cov))
            lM = np.log10(np.mean(10**lM_bins,1))

            #Create the model object
            NM_model_obj = NM_model_module.MF_model(cosmo_dict,bounds,volume,redshift)

            #Evaluate the model
            best_model = [1.97,1.0,f_test,g_test]
            NM_best = NM_model_obj.MF_model_all_bins(lM_bins,best_model,redshift)
            cov_fg = param_cov[0,2]+(pivot-sf)*(param_cov[1,2]+param_cov[0,3])+(pivot-sf)**2*param_cov[1,3]
            NM_var = NM_model_obj.var_MF_model_all_bins(lM_bins,best_model,[f_var,g_var])#,cov_fg)
            NM_best_err = np.sqrt(NM_var)
            
            #for j in range(len(lM)):
            #    print lM_bins[j],"%e +- %e    %e+-%e"%(NM_data[j],NM_err[j],NM_best[j],NM_best_err[j])

            #visualize.NM_plot(lM,NM_data,NM_err,lM,NM_best)
            title = "Box%03d left out for z=%f"%(i,redshift)
            savepath = "plots/NM_plots/NM_LOO_emulated_Box%03d_Z%d.png"%(i,z_index)
            sigma_savepath = "plots/gsigma_plots/gsigma_LOO_emulated_Box%03d_Z%d.png"%(i,z_index)
            visualize.NM_emulated(lM,NM_data,NM_err,lM,NM_best,NM_best_err,title,savepath)
            visualize.g_sigma_emulated(NM_model_obj,redshift,volume,cosmo_dict,lM,lM_bins,NM_data,NM_err,best_model,[f_var,g_var],title,sigma_savepath)
