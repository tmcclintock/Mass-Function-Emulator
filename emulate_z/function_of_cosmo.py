"""
This is a short script that creates full emulators and predicts 
f and g as functions of cosmology.
"""

import numpy as np
import Emulator
import sys

scale_factors = np.array([0.25,0.333333,0.5,0.540541,0.588235,0.645161,0.714286,0.8,0.909091,1.0])
redshifts = 1./scale_factors - 1.0
volume = 1.e9 #(1000.)**3 #(Mpc/h)^3

top_path = "/home/tmcclintock/Desktop/Github_stuff/Weak-Lensing-Emulator/"
path_to_cosmos = top_path+"cosmology_files/building_cosmos_no_z.txt"
cosmos = np.genfromtxt(path_to_cosmos).T
cosmos = np.delete(cosmos,39,1)
#Needs this shape to build the emulator from

param_names = [r"$\Omega_bh^2$",r"$\Omega_ch^2$",r"$w_0$",r"$n_s$",r"$\log_{10}A_s$",r"$H_0$",r"$N_{eff}$",r"$\sigma_8$"]
simple_param_names = ["Omegabh2","Omegach2","w0","ns","log10As","H0","Neff","sigma8"]

build_emulators = True
examine_emulators = False

f_name = "f_full_Z%d_emu"
g_name = "g_full_Z%d_emu"

for z_index in xrange(0,len(redshifts)):
    redshift = redshifts[z_index]
    means = np.genfromtxt(top_path+"z_fitting/chains/building_data/building_means_Z%d.txt"%z_index).T #means
    varis = np.genfromtxt(top_path+"z_fitting/chains/building_data/building_vars_Z%d.txt"%z_index).T #variances
    
    if build_emulators:
        fmean,gmean = means
        fvar,gvar = varis
        ferr,gerr = np.sqrt(fvar),np.sqrt(gvar)

        fmean_in = fmean #- np.mean(fmean)
        gmean_in = gmean #- np.mean(gmean)

        f_emu = Emulator.Emulator(name=f_name%(z_index),xdata=cosmos,ydata=fmean_in,yerr=ferr)
        g_emu = Emulator.Emulator(name=g_name%(z_index),xdata=cosmos,ydata=gmean_in,yerr=gerr)
        
        print "Training %s now"%(f_name%(z_index))
        f_emu.train()
        print "\t%s trained"%(f_name%(z_index))
        f_emu.save("saved_emulators/%s"%(f_name%(z_index)))
        print "\t%s saved"%(f_name%(z_index))
        print "Training %s now"%(g_name%(z_index))
        g_emu.train()
        print "\t%s trained"%(g_name%(z_index))
        g_emu.save("saved_emulators/%s"%(g_name%(z_index)))
        print "\t%s saved"%(g_name%(z_index))

    if examine_emulators:
        cosmos_mean = np.mean(cosmos,1)
        
        import matplotlib.pyplot as plt
        plt.rc('text',usetex=True, fontsize=20)

        space = np.ones_like(cosmos[:,0])
        f_emu = Emulator.Emulator(space,space,space)
        g_emu = Emulator.Emulator(space,space,space)
        f_emu.load("saved_emulators/%s"%(f_name%(z_index)))
        g_emu.load("saved_emulators/%s"%(g_name%(z_index)))
        print "Emulators loaded"

        for i in xrange(0,len(param_names)):
            print "Creating %s plot at Z%d"%(param_names[i],z_index)
            min_param = np.min(cosmos[i,:])
            max_param = np.max(cosmos[i,:])
            
            domain = np.linspace(min_param,max_param,100)

            f_model = np.zeros_like(domain)
            f_model_var = np.zeros_like(domain)

            g_model = np.zeros_like(domain)
            g_model_var = np.zeros_like(domain)
            for j in range(len(domain)):
                cosmos_here = np.copy(cosmos_mean)
                cosmos_here[i] = domain[j]
                f_model[j],f_model_var[j] = f_emu.predict_one_point(cosmos_here)
                g_model[j],g_model_var[j] = g_emu.predict_one_point(cosmos_here)
            f_model_err = np.sqrt(f_model_var)
            g_model_err = np.sqrt(g_model_var)

            f_model_upper = f_model + f_model_err
            f_model_lower = f_model - f_model_err
            g_model_upper = g_model + g_model_err
            g_model_lower = g_model - g_model_err

            
            fig,axarr = plt.subplots(2,sharex=True)
            axarr[0].set_title("Tinker params vs. %s at z=%0.2f"%(param_names[i],redshift))

            axarr[0].plot(domain,f_model,c='r')
            axarr[0].plot(domain,f_model_upper,c='g')
            axarr[0].plot(domain,f_model_lower,c='g')

            axarr[1].plot(domain,g_model,c='r')
            axarr[1].plot(domain,g_model_upper,c='g')
            axarr[1].plot(domain,g_model_lower,c='g')

            axarr[0].set_ylabel("f")
            axarr[1].set_ylabel("g")
            axarr[1].set_xlabel("%s"%param_names[i])
            plt.subplots_adjust(bottom=0.15,left=0.15,hspace=0.001)
            plt.gcf().savefig("plots/cosmo_plots/tinker_params_vs_%s_z%.2f.png"%(simple_param_names[i],redshift))
            #plt.show()
            plt.close()
