"""
Make the plots for my grad talk on 2/3.
"""
import mf_emulator
import emulator, sys
import numpy as np
import tinker_mass_function as TMF
sys.path.insert(0,'./visualization/')
import visualize
import matplotlib.pyplot as plt
plt.rc('text',usetex=True,fontsize=24)

savedir = "../../Presentations/grad_talk_spring_2017/"
datapath = "../../all_MF_data/building_MF_data/full_mf_data/Box%03d_full/Box%03d_full_Z%d.txt"
covpath = "../../all_MF_data/building_MF_data/covariances/Box%03d_cov/Box%03d_cov_Z%d.txt"
volume = 1050.**3 #[Mpc/h]^3

doscatter = True
showmf = True
showzs = True


#Read in the input cosmologies
all_cosmologies = np.genfromtxt("./test_data/building_cosmos_all_params.txt")
#all_cosmologies = np.delete(all_cosmologies,5,1) #Delete ln10As
all_cosmologies = np.delete(all_cosmologies,0,1) #Delete boxnum
all_cosmologies = np.delete(all_cosmologies,-1,0)#39 is broken
N_cosmologies = len(all_cosmologies)

if doscatter:
    ombh2 = all_cosmologies[:,0]
    omch2 = all_cosmologies[:,1]
    w = all_cosmologies[:,2]
    H0 = all_cosmologies[:,5]
    sigma8 = all_cosmologies[:,7]
    h = H0/100.
    Omb = ombh2/h**2
    Omc = omch2/h**2
    Om = Omb + Omc
    print Om[0],H0[0],w[0],sigma8[0]
    plt.scatter(H0,Om,s=30,c='k')
    plt.xlabel(r"$H_0\ [{\rm km/s/Mpc}]$")
    plt.ylabel(r"$\Omega_{\rm M}$")
    plt.subplots_adjust(bottom=0.15,left=0.15)
    plt.gcf().savefig(savedir+"h_vs_Om.png")
    plt.show()
    plt.clf()
    plt.scatter(w,sigma8,s=30,c='k')
    plt.xlabel(r"$w$")
    plt.ylabel(r"$\sigma_8$")
    plt.subplots_adjust(bottom=0.15,left=0.15)
    plt.gcf().savefig(savedir+"w_vs_s8.png")
    plt.show()
    plt.clf()

if showmf:
    #Read in the test mass function
    box = 0
    zind = 9
    MF_data = np.genfromtxt(datapath%(box,box,zind))
    lM_bins = MF_data[:,:2]
    N = MF_data[:,2]
    cov_data = np.genfromtxt(covpath%(box,box,zind))
    N_err = np.sqrt(np.diagonal(cov_data))
    lM = np.log10(np.mean(10**lM_bins,1))

    fig,ax = plt.subplots(1,1)
    ax.errorbar(lM,N,yerr=N_err,marker='.',color='k')
    ax.set_yscale('log')
    ax.set_xlabel(r"$\log_{10}M\ [{\rm M_\odot}/h]$")
    ylims = ax.get_ylim()
    ax.set_xlim(12.9,15.7)
    ax.set_ylim(1e-1,ylims[1])
    ax.set_ylabel(r"${\rm Number}/[1\ {\rm Gpc^3} \log_{10}{\rm M_\odot}]$")
    plt.subplots_adjust(bottom=0.15,left=0.15,hspace=0.001)
    fig.savefig(savedir+"MF_example.png")
    plt.show()
    plt.clf()

if showzs:
    box = 0
    Nz = 10
    fig,ax = plt.subplots(1,1)
    for zind in xrange(0,Nz):
        MF_data = np.genfromtxt(datapath%(box,box,zind))
        lM_bins = MF_data[:,:2]
        N = MF_data[:,2]
        cov_data = np.genfromtxt(covpath%(box,box,zind))
        N_err = np.sqrt(np.diagonal(cov_data))
        volume = 1050.**3 #[Mpc/h]^3
        lM = np.log10(np.mean(10**lM_bins,1))
        ax.errorbar(lM,N,yerr=N_err,color=plt.cm.seismic(1.-1.*zind/(Nz-1)),marker='.')
    ax.set_yscale('log')
    ax.set_xlabel(r"$\log_{10}M\ [{\rm M_\odot}/h]$")
    ylims = ax.get_ylim()
    ax.set_ylim(1e-1,ylims[1])
    ax.set_xlim(12.9,15.7)
    ax.set_ylabel(r"${\rm Number}/[1\ {\rm Gpc^3} \log_{10}{\rm M_\odot}]$")
    plt.subplots_adjust(bottom=0.15,left=0.15,hspace=0.001)
    fig.savefig(savedir+"MF_reds.png")
    plt.show()
    plt.clf()
    
