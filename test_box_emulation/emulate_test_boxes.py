"""
This script emulates the test boxes.
"""
import numpy as np
import sys
sys.path.insert(0,"../NM_model/")
sys.path.insert(0,"../Emulator/")
sys.path.insert(0,"../visualization/")
import Emulator
import NM_model as NM_model_module
import visualize

Ntests = 7
Nreals = 5
Nreds = 10

scale_factors = np.array([0.25,0.333333,0.5,0.540541,0.588235,0.645161,0.714286,0.8,0.909091,1.0])
redshifts = 1./scale_factors - 1.0
volume = 1.e9 #(1000.)**3 #(Mpc/h)^3

boxname = "TestBox%03d-%03d"

data_base = "/home/tmcclintock/Desktop/Test_NM_data/"
data_path = data_base+"full_mf_data/%s_full/%s_full_Z%d.txt"
cov_path = data_base+"/covariances/%s_cov/%s_cov_Z%d.txt"

cosmos_path = "../cosmology_files/test_cosmologies.txt"
cosmos = np.genfromtxt(cosmos_path)

"""
Loop over each test box
"""
f_name = "f_full_Z%d_emu"
g_name = "g_full_Z%d_emu"

for box_ind in xrange(0,1):#Ntests):
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
    
    for z_index in xrange(9,Nreds):
        redshift = redshifts[z_index]
        space = np.ones_like(cosmo) #A filler array
        f_emu = Emulator.Emulator(space,space,space)
        g_emu = Emulator.Emulator(space,space,space)
        f_emu.load("saved_emulators/%s"%(f_name%(z_index)))
        g_emu.load("saved_emulators/%s"%(g_name%(z_index)))
        print "Emulators loaded"
        f_test,f_var = f_emu.predict_one_point(cosmo)
        g_test,g_var = g_emu.predict_one_point(cosmo)
        
        for real_ind in xrange(Nreals-1,Nreals):
            
            box = boxname%(box_ind,real_ind)
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
            print f_test, np.sqrt(f_var)
            print g_test, np.sqrt(g_var)
            NM_best = NM_model_obj.MF_model_all_bins(lM_bins,best_model,redshift)
            NM_var = NM_model_obj.var_MF_model_all_bins(lM_bins,best_model,[f_var,g_var])
            NM_best_err = np.sqrt(NM_var)

            #visualize.NM_plot(lM,NM_data,NM_err,lM,NM_best)
            title = "%s at z=%f"%(box,redshift)
            savepath = "plots/NM_plots/NM_emulated_%s_Z%d.png"%(box,z_index)
            sigma_savepath = "plots/gsigma_plots/gsigma_emulated_%s_Z%d.png"%(box,z_index)
            
            print "\tcreating NM plot for %s at Z%d or z=%.2f"%(box,z_index,redshift)
            visualize.NM_emulated(lM,NM_data,NM_err,lM,NM_best,NM_best_err,title,savepath)
            #print "\tcreating g(sigma) plot for %s at Z%d or z=%.2f"%(box,z_index,redshift)
            #visualize.g_sigma_emulated(NM_model_obj,redshift,volume,cosmo_dict,lM,lM_bins,NM_data,NM_err,best_model,[f_var,g_var],title,sigma_savepath)