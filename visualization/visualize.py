import numpy as np
import sys
sys.path.insert(0,"../NM_model/")
import NM_model as NM_model_module
import cosmocalc as cc
import matplotlib.pyplot as plt
plt.rc('text',usetex=True, fontsize=20)

show_plots = True

def single_NM_plot(lM_data, NM_data, NM_err):
    fig,ax = plt.subplots(1,1)
    
    ax.errorbar(lM_data,NM_data,yerr=NM_err)
    ax.set_yscale('log')

    ax.set_xlabel(r"$\log_{10}M\ [M_\odot/h]$")
    ylims = ax.get_ylim()
    ax.set_ylim(1e-1,ylims[1])
    ax.set_ylabel(r"$N(M,z)$")
    plt.subplots_adjust(bottom=0.15,left=0.15,hspace=0.001)
    if show_plots:
        plt.show()
    plt.close()



def NM_plot(lM_data, NM_data, NM_err, lM_model, NM_model):
    f,axarr = plt.subplots(2, sharex = True)
    
    axarr[0].errorbar(lM_data,NM_data,yerr=NM_err)
    axarr[0].plot(lM_model,NM_model)
    axarr[0].set_yscale('log')

    pdiff = 100*(NM_data - NM_model)/NM_model
    axarr[1].errorbar(lM_data,pdiff,100*NM_err/NM_model)
    axarr[1].set_ylim(-50,50)
    axarr[1].set_xlabel(r"$\log_{10}M\ [M_\odot/h]$")
    axarr[0].set_ylabel(r"$N(M,z)$")
    axarr[1].set_ylabel(r"$\%$ Diff")
    plt.subplots_adjust(bottom=0.15,left=0.15,hspace=0.001)
    if show_plots:
        plt.show()
    plt.close()

def NM_emulated(lM_data, NM_data, NM_err, lM_emu, NM_emu, NM_emu_err,title,savepath):
    f,axarr = plt.subplots(2, sharex = True)
    axarr[0].set_title(title)

    NM_upper = NM_emu + NM_emu_err
    NM_lower = NM_emu - NM_emu_err
    
    axarr[0].errorbar(lM_data,NM_data,yerr=NM_err)
    axarr[0].plot(lM_emu,NM_emu,c='r')
    axarr[0].plot(lM_emu,NM_upper,c='g')
    axarr[0].plot(lM_emu,NM_lower,c='g')
    axarr[0].set_yscale('log')

    #pdiff = 100*(NM_data - NM_emu)/NM_emu
    #pdiff_upper = 100*(NM_data - NM_upper)/NM_upper
    #pdiff_lower = 100*(NM_data - NM_lower)/NM_lower
    #axarr[1].errorbar(lM_data,pdiff,100*NM_err/NM_emu)
    #axarr[1].plot(lM_data,pdiff_upper,c="g")
    #axarr[1].plot(lM_data,pdiff_lower,c="g")

    resid = (NM_data - NM_emu)/np.sqrt(((0.01*NM_emu)**2+NM_err**2))
    resid_err = NM_err/np.sqrt(((0.01*NM_emu)**2+NM_err**2))
    resid_upper = (NM_data - NM_upper)/np.sqrt(((0.01*NM_upper)**2+NM_err**2))
    resid_lower = (NM_data - NM_lower)/np.sqrt(((0.01*NM_lower)**2+NM_err**2))

    axarr[1].errorbar(lM_data,resid,resid_err,c='b')
    axarr[1].plot(lM_data,resid_upper,c='g')
    axarr[1].plot(lM_data,resid_lower,c='g')

    #Draw dashed lines at 1%
    lims = axarr[1].get_xlim()
    axarr[1].plot(lims,[1,1],"k--",zorder=-1,alpha=0.5)
    axarr[1].plot(lims,[-1,-1],"k--",zorder=-1,alpha=0.5)

    #axarr[1].set_ylim(-50,50)
    axarr[1].set_ylim(-5,5)
    axarr[1].set_xlabel(r"$\log_{10}M\ [M_\odot/h]$")
    axarr[0].set_ylabel(r"$N(M,z)$")
    #axarr[1].set_ylabel(r"$\%$ Diff")
    axarr[1].set_ylabel(r"$\frac{N-N_m}{\sqrt{(0.01N_m)^2+\Delta N^2}}$")

    plt.subplots_adjust(bottom=0.15,left=0.15,hspace=0.001)
    plt.gcf().savefig(savepath)
    if show_plots:
        plt.show()
    plt.clf()
    plt.close()

def g_sigma_plot(NM_model_obj,redshift,volume,cosmo_dict,\
                     lM_data,lM_bins,NM_data,NM_err,best_params):
    G = 4.52e-48 #Newton's gravitional constant in Mpc^3/s^2/Solar Mass
    Mpcperkm = 3.241e-20 #Mpc/km; used to convert H0 to s^-1
    Om,H0 = cosmo_dict["om"],cosmo_dict["h"]*100.0
    rhom=Om*3.*(H0*Mpcperkm)**2/(8*np.pi*G*(H0/100.)**2)#Msun h^2/Mpc^3

    scale_factor = 1./(1+redshift)
    M = 10**lM_data
    dM = 1e-6*M
    bin_widths = 10**lM_bins[:,1] - 10**lM_bins[:,0]
    #First, we figure out how the bins change into sigmas
    #small sigma is large mass
    sigma = np.ones_like(lM_data)
    dlnsigdm = np.ones_like(lM_data)
    g_model = np.ones_like(lM_data)
    for i in range(len(lM_data)):
        sigma[i] = cc.sigmaMtophat_exact(10**lM_data[i],scale_factor)
        dlnsigdm[i] = np.log(cc.sigmaMtophat_exact(M[i]-dM[i]/2,scale_factor)/cc.sigmaMtophat_exact(M[i]+dM[i]/2,scale_factor))/dM[i]
        g_model[i] = NM_model_obj.calc_g(sigma[i],best_params)

    g_sigma = NM_data/(bin_widths*volume*rhom/M*dlnsigdm)
    gs_err = NM_err/(bin_widths*volume*rhom/M*dlnsigdm)

    pdiff = 100*(g_sigma - g_model)/g_model
    pdiff_err = 100*gs_err/g_model

    f,axarr = plt.subplots(2, sharex = True)

    axarr[0].errorbar(sigma,g_sigma,gs_err)
    axarr[0].plot(sigma,g_model)
    axarr[1].errorbar(sigma,pdiff,yerr=pdiff_err)

    axarr[0].set_yscale('log')
    axarr[0].set_ylim(1e-5,1)
    axarr[1].set_ylim(-50,50)
    axarr[1].set_xlabel(r"$\sigma$")
    axarr[0].set_ylabel(r"$g(\sigma)$")
    axarr[1].set_ylabel(r"$\%$ Diff")
    plt.subplots_adjust(bottom=0.15,left=0.15,hspace=0.001)
    if show_plots:
        plt.show()

def g_sigma_emulated(NM_model_obj,redshift,volume,cosmo_dict,lM_data,lM_bins,NM_data,NM_err,best_params,variances,title,savepath):
    G = 4.52e-48 #Newton's gravitional constant in Mpc^3/s^2/Solar Mass
    Mpcperkm = 3.241e-20 #Mpc/km; used to convert H0 to s^-1
    Om,H0 = cosmo_dict["om"],cosmo_dict["h"]*100.0
    rhom=Om*3.*(H0*Mpcperkm)**2/(8*np.pi*G*(H0/100.)**2)#Msun h^2/Mpc^3

    scale_factor = 1./(1+redshift)
    M = 10**lM_data
    dM = 1e-6*M
    bin_widths = 10**lM_bins[:,1] - 10**lM_bins[:,0]
    #First, we figure out how the bins change into sigmas
    #small sigma is large mass
    sigma = np.ones_like(lM_data)
    dlnsigdm = np.ones_like(lM_data)
    g_model = np.ones_like(lM_data)
    g_model_err = np.ones_like(lM_data)

    for i in range(len(lM_data)):
        sigma[i] = cc.sigmaMtophat_exact(10**lM_data[i],scale_factor)
        dlnsigdm[i] = np.log(cc.sigmaMtophat_exact(M[i]-dM[i]/2,scale_factor)/cc.sigmaMtophat_exact(M[i]+dM[i]/2,scale_factor))/dM[i]
        g_model[i] = NM_model_obj.calc_g(sigma[i],best_params)
        g_model_err[i] = np.sqrt(variances[0]*NM_model_obj.dg_df(sigma[i],best_params)**2+variances[1]*NM_model_obj.dg_dg(sigma[i],best_params)**2)

    g_sigma = NM_data/(bin_widths*volume*rhom/M*dlnsigdm)
    gs_err = NM_err/(bin_widths*volume*rhom/M*dlnsigdm)

    g_model_upper = g_model+g_model_err
    g_model_lower = g_model-g_model_err

    pdiff = 100*(g_sigma - g_model)/g_model
    pdiff_err = 100*gs_err/g_model
    pdiff_upper = 100*(g_sigma - g_model_upper)/g_model_upper
    pdiff_lower = 100*(g_sigma - g_model_lower)/g_model_lower

    f,axarr = plt.subplots(2, sharex = True)
    axarr[0].set_title(title)

    axarr[0].errorbar(sigma,g_sigma,gs_err,c='b')
    axarr[0].plot(sigma,g_model,c='r')
    axarr[0].plot(sigma,g_model_upper,c='g')
    axarr[0].plot(sigma,g_model_lower,c='g')

    #axarr[1].errorbar(sigma,pdiff,yerr=pdiff_err)
    #axarr[1].plot(sigma,pdiff_upper,c='g')
    #axarr[1].plot(sigma,pdiff_lower,c='g')

    resid = (g_sigma - g_model)/np.sqrt(((0.01*g_model)**2+gs_err**2))
    resid_err = gs_err/np.sqrt(((0.01*g_model)**2+gs_err**2))
    resid_upper = (g_sigma - g_model_upper)/np.sqrt(((0.01*g_model_upper)**2+gs_err**2))
    resid_lower = (g_sigma - g_model_lower)/np.sqrt(((0.01*g_model_lower)**2+gs_err**2))

    axarr[1].errorbar(sigma,resid,resid_err,c='b')
    axarr[1].plot(sigma,resid_upper,c='g')
    axarr[1].plot(sigma,resid_lower,c='g')

    axarr[0].set_yscale('log')
    axarr[0].set_ylim(1e-5,1)
    axarr[1].set_ylim(-5,5)
    axarr[1].set_xlabel(r"$\sigma$")
    axarr[0].set_ylabel(r"$g(\sigma)$")
    #axarr[1].set_ylabel(r"$\%$ Diff")

    axarr[1].set_ylabel(r"$\frac{g(\sigma)-g_m(\sigma)}{\sqrt{(0.01g_m)^2+\Delta g^2}}$")

    plt.subplots_adjust(bottom=0.15,left=0.15,hspace=0.001)
    plt.gcf().savefig(savepath)
    if show_plots:
        plt.show()
    plt.clf()
    plt.close()
