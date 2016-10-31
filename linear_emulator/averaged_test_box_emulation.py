"""
This script emulates the test boxes with the linear model.
"""
import numpy as np
import sys
sys.path.insert(0,"../NM_model/")
sys.path.insert(0,"../Emulator/")
sys.path.insert(0,"../visualization/")
import visualize
import Emulator
import NM_model as NM_model_module

Ntests = 7
Nreals = 5
Nreds = 10

scale_factors = np.array([0.25,0.333333,0.5,0.540541,0.588235,\
                              0.645161,0.714286,0.8,0.909091,1.0])
redshifts = 1./scale_factors - 1.0
volume = 1.e9 #(1000.)**3 #(Mpc/h)^3

boxname = "TestBox%03d"

data_base = "/home/tmcclintock/Desktop/Test_NM_data/averaged_mf_data/"
data_path = data_base+"/full_mf_data/%s/%s_mean_Z%d.txt"
cov_path = data_base+"/covariances/%s_cov/%s_cov_Z%d.txt"

cosmos_path = "../cosmology_files/test_cosmologies.txt"
cosmos = np.genfromtxt(cosmos_path)

#This is the scale factor pivot
pivot = 0.5

"""
Read in the emulators.
"""
f0_name = "f0_emu"
g0_name = "g0_emu"
f1_name = "f1_emu"
g1_name = "g1_emu"
space = np.ones_like(cosmos[:,0]) #A filler array
f0_emu = Emulator.Emulator(space,space,space)
g0_emu = Emulator.Emulator(space,space,space)
f1_emu = Emulator.Emulator(space,space,space)
g1_emu = Emulator.Emulator(space,space,space)
f0_emu.load("saved_emulators/%s"%f0_name)
g0_emu.load("saved_emulators/%s"%g0_name)
f1_emu.load("saved_emulators/%s"%f1_name)
g1_emu.load("saved_emulators/%s"%g1_name)

#Read in the hyperparameter covariances
param_cov = np.loadtxt("../building_data/mcmc_tinker_hyperparam_covariances_v2.txt")
#param_cov = np.loadtxt("../building_data/tinker_hyperparam_covariances.txt")

"""
Loop over each test box.
"""
for box_ind in xrange(0,Ntests):
    cosmo = cosmos[box_ind]
    #Build a cosmo_dict
    h = cosmo[5]/100.
    Om = (cosmo[0]+cosmo[1])/h**2
    Ob = cosmo[0]/h**2
    sigma8 = cosmo[7]
    w0 = cosmo[2]
    ns = cosmo[3]
    cosmo_dict = {"om":Om,"ob":Ob,"ol":1-Om,\
                      "ok":0.0,"h":h,"s8":sigma8,\
                      "ns":ns,"w0":w0,"wa":0.0}
    f0_test,f0_var = f0_emu.predict_one_point(cosmo)
    g0_test,g0_var = g0_emu.predict_one_point(cosmo)
    f1_test,f1_var = f1_emu.predict_one_point(cosmo)
    g1_test,g1_var = g1_emu.predict_one_point(cosmo)

    #TRUE USING ONE LESS PARAMETER
    #Calculated from an independent program
    #slope,intercept = 0.386547512282,1.0072151586 #old fits
    slope,intercept = 0.278825271449, 1.05444508739
    g0_test = f0_test*slope + intercept
    g0_var  = f0_var*slope**2

    for z_index in xrange(0,Nreds):
        redshift = redshifts[z_index]
        sf = 1./(1.+redshift)
        #Predict f and g
        f_test = f0_test + (pivot-sf)*f1_test
        f_var = f0_var + (pivot-sf)**2*f1_var
        g_test = g0_test + (pivot-sf)*g1_test
        g_var = g0_var + (pivot-sf)**2*g1_var

        cov_fg = slope*param_cov[0,0] + slope*(pivot-sf)*param_cov[0,1] + (pivot-sf)*param_cov[0,2] + (pivot-sf)**2*param_cov[1,2]
        cov_ff = param_cov[0,0] + 2*(pivot-sf)*param_cov[0,1] + (pivot-sf)**2*param_cov[1,1]
        cov_gg = slope**2*param_cov[0,0] + 2*slope*(pivot-sf)*param_cov[0,2] + (pivot-sf)**2*param_cov[2,2]

        box = boxname%(box_ind)
        data_test = data_path%(box,box,z_index)
        cov_test = cov_path%(box,box,z_index)
        print "Working with %s at Z%d or z=%.2f"%(box,z_index,redshift)
        print "\tCreating plots"

        #Read in the NM data
        indata = np.loadtxt(data_test)
        cov = np.loadtxt(cov_test)
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
        cov_emu = NM_model_obj.covariance_MF(lM_bins,best_model,[cov_ff,cov_gg],cov_fg)
        NM_best_err = np.sqrt(np.diagonal(cov_emu))

        #visualize.NM_plot(lM,NM_data,NM_err,lM,NM_best)
        title = "%s at z=%.2f"%(box,redshift)
        savepath = "plots/NM_plots/NM_emulated_ave_%s_Z%d.png"%(box,z_index)
        sigma_savepath = "plots/gsigma_plots/gsigma_emulated_ave_%s_Z%d.png"%(box,z_index)
            
        print "\tcreating NM plot for %s at Z%d or z=%.2f"%(box,z_index,redshift)
        visualize.NM_emulated(lM,NM_data,NM_err,lM,NM_best,NM_best_err,title,savepath)
        print "\tcreating g(sigma) plot for %s at Z%d or z=%.2f"%(box,z_index,redshift)
        #visualize.g_sigma_emulated(NM_model_obj,redshift,volume,cosmo_dict,lM,lM_bins,NM_data,NM_err,best_model,[f_var,g_var],title,sigma_savepath)
