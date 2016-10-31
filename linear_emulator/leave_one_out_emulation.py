"""
This builds the leave-one-out emulators.
"""
import numpy as np
import os,sys
import matplotlib as m
import matplotlib.pyplot as plt
sys.path.insert(0,"../visualization/")
import emulator as Emulator
import tinker_mass_function as TMF
import visualize

Ncosmos = 39 #40th is broken
Nreds = 10

scale_factors = np.array([0.25,0.333333,0.5,0.540541,0.588235,0.645161,0.714286,0.8,0.909091,1.0])
redshifts = 1./scale_factors - 1.0
volume = 1.e9 #(1000.)**3 #(Mpc/h)^3

data_base = "/home/tmcclintock/Desktop/Mass_Function_Data/"#BACKUP_20Mbins/"
data_path = data_base+"/full_mf_data/Box%03d_full/Box%03d_full_Z%d.txt"
cov_data_path = data_base+"/covariances/Box%03d_cov/Box%03d_cov_Z%d.txt"

build_emulators = False
examine_emulators = True
output_derivs = False
view_all_corrs = False
visualize_curves = True
output_emulator_predictions = False

path_to_cosmos = "../cosmology_files/building_cosmos_no_z.txt"
cosmos = np.genfromtxt(path_to_cosmos)
cosmos = np.delete(cosmos,39,0)
#Needs this shape to build the emulator from

#Output for derivatives
dNdf_path = "covariance_matrix/deriv_txt_files/f_deriv_cosmo%d_Z%d.txt"
dNdg_path = "covariance_matrix/deriv_txt_files/g_deriv_cosmo%d_Z%d.txt"

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

#Keep arrays with the predictions
f0_predicted = []
f1_predicted = []
g0_predicted = []
g1_predicted = []
f0_var_predicted = []
f1_var_predicted = []
g0_var_predicted = []
g1_var_predicted = []


def view_corr(cov,title,labels=None):
    corr = np.zeros_like(cov)
    for i in range(len(cov)):
        for j  in range(len(cov[i])):
            corr[i,j] = cov[i,j]/np.sqrt(cov[i,i]*cov[j,j])
    plt.pcolor(np.flipud(corr),vmin=-1.0,vmax=1.0)
    if labels is not None:
        ax = plt.gca()
        ax.set_xticks(np.arange(len(cov))+0.5,minor=False)
        ax.set_xticklabels(labels,minor=False)
        ax.set_yticks(np.arange(len(cov))+0.5,minor=False)
        ax.set_yticklabels(reversed(labels),minor=False)
    plt.title(title)
    plt.colorbar()
    plt.show()

"""
Loop over each cosmology and
build an emulator without that cosmology included.

Then save the emulator.
"""

for i in xrange(36,37):#,Ncosmos,1):
    if build_emulators:
        cosmos_used = np.delete(cosmos,i,0)
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
        cosmo_predicted = cosmos[i]
        print cosmo_predicted
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

        print f0_real,g0_real,f1_real,g1_real
        print f0_test,g0_test,f1_test,g1_test

        f0_predicted.append(f0_test)
        f1_predicted.append(f1_test)
        g0_predicted.append(g0_test)
        g1_predicted.append(g1_test)
        f0_var_predicted.append(f0_var)
        f1_var_predicted.append(f1_var)
        g0_var_predicted.append(g0_var)
        g1_var_predicted.append(g1_var)

        #TRUE USING ONE LESS PARAMETER
        #Calculated from an independent program
        #slope,intercept = 0.386547512282,1.0072151586 #old fits
        slope,intercept = 0.278825271449, 1.05444508739
        g0_test = f0_test*slope + intercept
        g0_var  = f0_var*slope**2

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
        param_cov = np.loadtxt("../building_data/mcmc_tinker_hyperparam_covariances_v2.txt")

        for z_index in xrange(5,6):#,Nreds,1):
            print "\tExamining at Z%d"%z_index
            redshift = redshifts[z_index]
            sf = 1./(1.+redshift)
            f_test = f0_test + (pivot-sf)*f1_test
            f_var = f0_var + (pivot-sf)**2*f1_var 
            g_test = g0_test + (pivot-sf)*g1_test
            g_var = g0_var + (pivot-sf)**2*g1_var

            #Read in the NM data
            indata = np.loadtxt(data_path%(i,i,z_index))
            cov_data = np.loadtxt(cov_data_path%(i,i,z_index))
            lM_low,lM_high = indata[:,0],indata[:,1]
            lM_bins = np.array([lM_low,lM_high]).T
            NM_data = indata[:,2]
            
            #Remove bad data
            good_indices = np.where(NM_data > 0)[0]
            lM_bins = lM_bins[good_indices]
            bounds = np.array([np.min(lM_bins),np.max(lM_bins)])
            NM_data = NM_data[good_indices]
            cov_data = cov_data[:,good_indices]
            cov_data = cov_data[good_indices,:]
            NM_err = np.sqrt(np.diagonal(cov_data))
            lM = np.log10(np.mean(10**lM_bins,1))

            #Create the model object
            NM_model_obj = TMF.MF_model(cosmo_dict,redshift)
            NM_model_obj.set_parameters(1.97,1.0,f_test,g_test)

            #Evaluate the model
            n_model = NM_model_obj.n_in_bins(lM_bins)
            NM_model = n_model*volume

            #Get out the derivatives
            derivs =NM_model_obj.derivs_in_bins(lM_bins)
            dNdf_derivs = derivs[0,:]
            dNdg_derivs = derivs[1,:]
            if output_derivs:
                np.savetxt(dNdf_path%(i,z_index),dNdf_derivs)
                np.savetxt(dNdg_path%(i,z_index),dNdg_derivs)

            cov_fg = slope*param_cov[0,0] + slope*(pivot-sf)*param_cov[0,1] + (pivot-sf)*param_cov[0,2] + (pivot-sf)**2*param_cov[1,2]
            cov_ff = param_cov[0,0] + 2*(pivot-sf)*param_cov[0,1] + (pivot-sf)**2*param_cov[1,1]
            cov_gg = slope**2*param_cov[0,0] + 2*slope*(pivot-sf)*param_cov[0,2] + (pivot-sf)**2*param_cov[2,2]
            fg_cov = np.zeros((2,2))
            fg_cov[0,0] = cov_ff
            fg_cov[1,1] = cov_gg
            fg_cov[0,1] = fg_cov[1,0] = cov_fg

            #Save the best fit model
            os.system("mkdir -p txt_files/LOO_best_models/Box%03d/"%i)
            np.savetxt("txt_files/LOO_best_models/Box%03d/LOO_best_Box%03d_Z%d.txt"%(i,i,z_index),NM_model)

            #Save the emulator covariance matrix
            cov_emu = NM_model_obj.covariance_in_bins(lM_bins,[cov_ff,cov_gg],cov_fg)*volume**2
            NM_model_err = np.sqrt(np.diagonal(cov_emu))
            os.system("mkdir -p txt_files/LOO_covariance_matrices/Box%03d/"%i)
            np.savetxt("txt_files/LOO_covariance_matrices/Box%03d/LOO_cov_Box%03d_Z%d.txt"%(i,i,z_index),cov_emu)

            def my_chi2(N,cov_data,cov_model,NM_data,NM_model):
                cov_total = cov_data + cov_model
                diff = NM_data-NM_model
                icov = np.linalg.inv(cov_total)
                return np.dot(diff,np.dot(icov,diff))
            
            chi2 = my_chi2(len(NM_model),cov_data,cov_emu,NM_data,NM_model)
            
            if view_all_corrs:
                view_corr(param_cov,"param corr",["$f_0$","$f_1$","$g_1$"])
                view_corr(fg_cov,"fg corr",["$f$","$g$"])
                view_corr(cov,"data corr")
                view_corr(cov_emu,"emu corr")
            
            #Title for plots
            title = r"Box%03d left out at z=%.2f with $\chi^2_{dof=%d}=%.2f$"%(i,redshift,len(NM_model),chi2)

            if visualize_curves:
                savepath = "plots/NM_plots/NM_LOO_emulated_Box%03d_Z%d.png"%(i,z_index)
                sigma_savepath = "plots/gsigma_plots/gsigma_LOO_emulated_Box%03d_Z%d.png"%(i,z_index)
                visualize.NM_emulated(lM,NM_data,NM_err,lM,NM_model,NM_model_err,title,savepath)
                #visualize.g_sigma_emulated(NM_model_obj,redshift,volume,cosmo_dict,lM,lM_bins,NM_data,NM_err,best_model,[f_var,g_var],title,sigma_savepath)

if output_emulator_predictions:
    np.savetxt("txt_files/emulator_outputs/f0_predicted.txt",f0_predicted)
    np.savetxt("txt_files/emulator_outputs/f1_predicted.txt",f1_predicted)
    np.savetxt("txt_files/emulator_outputs/g0_predicted.txt",g0_predicted)
    np.savetxt("txt_files/emulator_outputs/g1_predicted.txt",g1_predicted)
    np.savetxt("txt_files/emulator_outputs/f0_var_predicted.txt",f0_var_predicted)
    np.savetxt("txt_files/emulator_outputs/f1_var_predicted.txt",f1_var_predicted)
    np.savetxt("txt_files/emulator_outputs/g0_var_predicted.txt",g0_var_predicted)
    np.savetxt("txt_files/emulator_outputs/g1_var_predicted.txt",g1_var_predicted)
