"""
This contains the code that builds the emulator and then saves it
in order to be used elsewhere.

This particular script is being used for the leave-one-out plots,
and so it builds 40 emulators each with a cosmology left out.
"""

import numpy as np
import sys
sys.path.insert(0,"../Emulator/")
sys.path.insert(0,"../NM_model/")
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

"""
Loop over each cosmology and
build an emulator without that cosmology included.

Then save the emulator.
"""
f_name = "f_%03d_Z%d_emu"
g_name = "g_%03d_Z%d_emu"

for z_index in xrange(9,len(redshifts)):
    means = np.genfromtxt("../building_data/building_means_Z%d.txt"%z_index).T #means
    varis = np.genfromtxt("../building_data/building_vars_Z%d.txt"%z_index).T #variances

    for i in xrange(0,1):#Ncosmos):
        cosmos_used = np.delete(cosmos,i,1)
        means_used = np.delete(means,i,1)
        varis_used = np.delete(varis,i,1)

        if build_emulators:
            fmean,gmean = means_used
            fvar,gvar = varis_used
            ferr,gerr = np.sqrt(fvar),np.sqrt(gvar)
            
            fmean_in = fmean #- np.mean(fmean)
            gmean_in = gmean #- np.mean(gmean)
            
            f_emu = Emulator.Emulator(name=f_name%(i,z_index),xdata=cosmos_used,ydata=fmean_in,yerr=ferr)
            g_emu = Emulator.Emulator(name=g_name%(i,z_index),xdata=cosmos_used,ydata=gmean_in,yerr=gerr)
            print "Training %s now"%(f_name%(i,z_index))
            f_emu.train()
            print "\t%s trained"%(f_name%(i,z_index))
            f_emu.save("saved_emulators/%s"%(f_name%(i,z_index)))
            print "\t%s saved"%(f_name%(i,z_index))
            print "Training %s now"%(g_name%(i,z_index))
            g_emu.train()
            print "\t%s trained"%(g_name%(i,z_index))
            g_emu.save("saved_emulators/%s"%(g_name%(i,z_index)))
            print "\t%s saved"%(g_name%(i,z_index))

        if examine_emulators:
            fmean,gmean = means_used

            means_real = means[:,i]
            err_real = np.sqrt(varis[:,i])
            
            cosmos_predicted = cosmos[:,i]
            
            space = np.ones_like(cosmos_predicted)
            f_emu = Emulator.Emulator(space,space,space)
            g_emu = Emulator.Emulator(space,space,space)
            f_emu.load("saved_emulators/%s"%(f_name%(i,z_index)))
            g_emu.load("saved_emulators/%s"%(g_name%(i,z_index)))
            print "Emulators loaded"
            f_test,f_var = f_emu.predict_one_point(cosmos_predicted)
            g_test,g_var = g_emu.predict_one_point(cosmos_predicted)
            f_test += 0 #np.mean(fmean)
            g_test += 0 #np.mean(gmean)
            
            print "Prediction complete"
            print "Predicted  std  Real err"
            print "%f %f %f %f"%(f_test,np.sqrt(f_var),means_real[0],err_real[0])
            print "%f %f %f %f"%(g_test,np.sqrt(g_var),means_real[1],err_real[1])
            
            print "Creating plots"
            
            sys.path.insert(0,'plot_routines/')
            sys.path.insert(0,'NM_model')

            
            #Build a cosmo_dict
            h = cosmos_predicted[5]/100.
            Om = (cosmos_predicted[0]+cosmos_predicted[1])/h**2
            Ob = cosmos_predicted[0]/h**2
            sigma8 = cosmos_predicted[7]
            w0 = cosmos_predicted[2]
            ns = cosmos_predicted[3]
            cosmo_dict = {"om":Om,"ob":Ob,"ol":1-Om,\
                              "ok":0.0,"h":h,"s8":sigma8,\
                              "ns":ns,"w0":w0,"wa":0.0}
            redshift = redshifts[z_index]
            
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
            NM_var = NM_model_obj.var_MF_model_all_bins(lM_bins,best_model,[f_var,g_var])
            NM_best_err = np.sqrt(NM_var)
            
            #for j in range(len(lM)):
            #    print lM_bins[j],"%e +- %e    %e+-%e"%(NM_data[j],NM_err[j],NM_best[j],NM_best_err[j])

            #visualize.NM_plot(lM,NM_data,NM_err,lM,NM_best)
            title = "Box%03d left out for z=%f"%(i,redshift)
            savepath = "plots/NM_plots/NM_LOO_emulated_Box%03d_Z%d.png"%(i,z_index)
            sigma_savepath = "plots/gsigma_plots/gsigma_LOO_emulated_Box%03d_Z%d.png"%(i,z_index)
            visualize.NM_emulated(lM,NM_data,NM_err,lM,NM_best,NM_best_err,title,savepath)
            #visualize.g_sigma_emulated(NM_model_obj,redshift,volume,cosmo_dict,lM,lM_bins,NM_data,NM_err,best_model,[f_var,g_var],title,sigma_savepath)
