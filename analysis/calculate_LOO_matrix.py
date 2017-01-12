"""
This script is used to calculate the parameter covariance matrix
need to minimize the chi2 fits for the emulator estimates. It does
not change the predictions at all from the emulator, but it 
does properly characterize the error coming out of the emulator.

The idea is as follows:
- create an Ensemble sampler with N_walkers walkers that will take N_steps steps
- For each box, pick a random redshift to sample at.
- Inside the likelihood, loop over each box-z pair and evaluate the N(M,z) fit
- The total chi2 is the sum over all the individual chi2s for each box-z pair
"""

import numpy as np
import scipy.optimize as op
import pickle, sys, os, emcee, corner, copy
import likelihoods_for_covariance as lhfuncs
import matplotlib.pyplot as plt

#Flow control
do_single_test = False
do_maximization = False
do_MCMC = False
do_analysis = False
average_chains = False
make_corrs = True

#MCMC information
N_trials = 100 #number of trials within a single MCMC step
N_dim = 10 #Number of free parameters
N_walkers = 3*N_dim #2*N_dim minimum
N_steps = 10000
N_burn = 5000

#Get in the pre-computed MF (N) and dNdp arrays
N_data_array = pickle.load(open("N_data_array.p","rb"))
cov_data_array = pickle.load(open("cov_data_array.p","rb"))
N_emu_array = pickle.load(open("N_emu_array.p","rb"))
dNdf_array = pickle.load(open("dNdf_array.p","rb"))
dNdg_array = pickle.load(open("dNdg_array.p","rb"))
dNdfxdNdf_array = pickle.load(open("dNdfxdNdf_array.p","rb"))
dNdgxdNdg_array = pickle.load(open("dNdgxdNdg_array.p","rb"))
dNdfxdNdg_array = pickle.load(open("dNdfxdNdg_array.p","rb"))

#Create the z_index array
N_cosmos = 39#Number of data files
N_z = 10#Number of redshifts

#Scale factors and redshifts
scale_factors = np.array([0.25,0.333333,0.5,0.540541,0.588235,
                          0.645161,0.714286,0.8,0.909091,1.0])
redshifts = 1./scale_factors - 1.0
k = scale_factors - 0.5 #f = f0 + k*f1

#Initial parameter guess
Vscale = 1e-3 #Scale of the variances
lVs = np.log(Vscale)
initial_parameter_guess = np.array((lVs,lVs,lVs,lVs,0.,0.,0.,0.,0.,0.)) #if using logs
#initial_parameter_guess = np.array((Vscale,Vscale,Vscale,Vscale,0.,0.,0.,0.,0.,0.)) #otherwise

#Create the redshift indices
from_scratch = False
if from_scratch:
    all_z_indices = np.random.randint(0,10,N_cosmos*N_trials)
    all_z_indices = all_z_indices.reshape(N_trials,N_cosmos)
    np.savetxt("output_files/all_z_indices.txt",all_z_indices)
else:
    all_z_indices = np.loadtxt("output_files/all_z_indices.txt")

#Perform various actions
lnprob = lhfuncs.lnprob
if do_single_test:
    z_indices = np.random.randint(0,N_z,N_cosmos)
    print lnprob(initial_parameter_guess,N_emu_array,\
                     N_data_array,cov_data_array,\
                     dNdfxdNdf_array,dNdgxdNdg_array,\
                     dNdfxdNdg_array,k,N_cosmos,N_z,z_indices)

if do_maximization:
    for trial in xrange(0,N_trials):
        z_indices = all_z_indices[trial]
        print "Maximizing Trial %d: "%trial
        print "Using: ",z_indices
        nll = lambda *args:-lnprob(*args)
        result = op.minimize(nll,initial_parameter_guess,\
                                 args=(N_emu_array,\
                                           N_data_array,cov_data_array,\
                                           dNdfxdNdf_array,dNdgxdNdg_array,\
                                           dNdfxdNdg_array,k,\
                                           N_cosmos,N_z,z_indices),\
                                 method="Powell")
        print result
        np.savetxt("output_files/maximization_result%d.txt"%trial,result['x'])

if do_MCMC:
    for trial in xrange(0,N_trials):
        start = np.loadtxt("output_files/maximization_result%d.txt"%trial)
        z_indices = all_z_indices[trial]
        print "\nOn trial %d, running MCMC on %d dimensions with %d walkers for %d steps"%(trial,N_dim,N_walkers,N_steps)
        pos = np.zeros((N_walkers,N_dim))
        for i in range(N_walkers):
            pos[i] += start
            pos[i,:4] += np.random.randn(4)*1e-1
            for j in xrange(4,N_dim):
                pos[i,j] = start[j] + np.random.randn()*1e-1
                if pos[i,j] > 1.0: pos[i,j] -= 1.0
                if pos[i,j] <-1.0: pos[i,j] += 1.0
                continue
            continue
        
        sampler_args = (N_emu_array,N_data_array,cov_data_array,dNdfxdNdf_array,dNdgxdNdg_array,dNdfxdNdg_array,k,N_cosmos,N_z,z_indices)
        sampler = emcee.EnsembleSampler(N_walkers,N_dim,lnprob,args=sampler_args)
        sampler.run_mcmc(pos,N_steps)
        full_chain = sampler.chain
        chain = full_chain[:,N_burn:].reshape(int(N_walkers*(N_steps-N_burn)),N_dim)
        likes = sampler.lnprobability
        np.savetxt("output_files/chains/chain_trial%d.txt"%trial,chain)
        np.save("output_files/chains/full_chain_trial%d"%trial,full_chain)
        np.savetxt("output_files/chains/likes_trial%d.txt"%trial,likes)

names = [r"$\lnC_{f_0,f_0}$",r"$\lnC_{f_1,f_1}$",r"$\lnC_{g_0,g_0}$",r"$\lnC_{g_1,g_1}$",\
             r"$R_{f_0,f_1}$",r"$R_{f_0,g_0}$",r"$R_{f_0,g_1}$",\
             r"$R_{f_1,g_0}$",r"$R_{f_1,g_1}$",r"$R_{g_0,g_1}$"]

if do_analysis:        
    for trial in range(N_trials):
        full_chain = np.load("output_files/chains/full_chain_trial%d.npy"%trial)
        print "Creating corner for trial %d"%trial
        chain = full_chain[:,N_burn:].reshape(int(N_walkers*(N_steps-N_burn)),N_dim)
        fig = corner.corner(chain,labels=names,plot_datapoints=False)
        fig.savefig("output_files/chains/fig_chain_trial%d.png"%trial)
        plt.close()

if average_chains:
    from_scratch = True
    if from_scratch:
        means = np.zeros((N_trials,N_dim)) #Holds all chain means
        var   = np.zeros((N_trials,N_dim)) #Holds all chain variances
        for trial in xrange(0,N_trials):
            full_chain = np.load("output_files/chains/full_chain_trial%d.npy"%trial)
            chain = full_chain[:,N_burn:].reshape(int(N_walkers*(N_steps-N_burn)),N_dim)
            means[trial] = np.mean(chain,0)
            var[trial]    = np.var(chain,0)
            continue
    else:
        means = np.loadtxt("output_files/chain_means.txt")
        var   = np.loadtxt("output_files/chain_vars.txt")
    np.savetxt("output_files/chain_means.txt",means)
    np.savetxt("output_files/chain_vars.txt",var)
    print "Full means:",np.mean(means,0)
    print "Full vars :",np.mean(var,0)
    slidenames = ["f0","f1","g0","g1","rf0f1","rf0g0","rf0g1","rf1g0","rf1g1","rg0g1"]
    for i in range(N_dim):
        plt.errorbar(np.arange(N_trials),means[:,i],yerr=np.sqrt(var[:,i]))
        plt.ylabel(names[i],fontsize=28)
        plt.xlabel("Trial number",fontsize=28)
        plt.subplots_adjust(bottom=0.15)
        plt.gcf().savefig("output_files/slide_plots/%s_vs_trial.png"%slidenames[i])
        plt.show()
        plt.close()

if make_corrs:
    def view_corr(cov,title):
        labels = [r"$f_0$",r"$f_1$",r"$g0$",r"$g_1$"]
        labels2 = [r"$f$",r"$g$"]

        corr = np.zeros_like(cov)
        for i in range(len(cov)):
            for j  in range(len(cov[i])):
                corr[i,j] = cov[i,j]/np.sqrt(cov[i,i]*cov[j,j])
        plt.pcolor(np.flipud(corr),vmin=-1.0,vmax=1.0)
        ax = plt.gca()
        if len(cov) == 4:
            ax.set_xticks(np.arange(len(cov))+0.5,minor=False)
            ax.set_xticklabels(labels,minor=False,fontsize=28)
            ax.set_yticks(np.arange(len(cov))+0.5,minor=False)
            ax.set_yticklabels(reversed(labels),minor=False,fontsize=28)
        else:
            ax.set_xticks(np.arange(len(cov))+0.5,minor=False)
            ax.set_xticklabels(labels2,minor=False,fontsize=28)
            ax.set_yticks(np.arange(len(cov))+0.5,minor=False)
            ax.set_yticklabels(reversed(labels2),minor=False,fontsize=28)
        plt.title(title)
        plt.colorbar()
        #plt.savefig("./figures/"+title+".png")
        plt.show()

    def get_corr_from_params(params):
        covs,rs = np.exp(params[:4]),params[4:]
        corr = np.diag((1.,1.,1.))
        corr[0,1] = corr[1,0] = rs[0]
        corr[0,2] = corr[2,0] = rs[1]
        corr[0,3] = corr[3,0] = rs[3]
        corr[1,2] = corr[2,1] = rs[4]
        corr[1,3] = corr[3,1] = rs[5]
        corr[2,3] = corr[3,2] = rs[6]
        return corr

    means = np.loadtxt("output_files/chain_means.txt")
    var   = np.loadtxt("output_files/chain_vars.txt")
    fullmeans = np.mean(means,0)
    Cf0f0,Cf1f1,Cg0g0,Cg1g1 = np.exp(fullmeans[:4])
    Rf0f1,Rf0g0,Rf0g1,Rf1g0,Rf1g1,Rg0g1 = fullmeans[4:]
    cov = np.diag(np.exp(fullmeans[:4]))
    cov[0,1] = cov[1,0] = Rf0f1*np.sqrt(Cf0f0*Cf1f1)
    cov[0,2] = cov[2,0] = Rf0g0*np.sqrt(Cf0f0*Cg0g0)
    cov[0,3] = cov[3,0] = Rf0g1*np.sqrt(Cf0f0*Cg1g1)
    cov[1,2] = cov[2,1] = Rf0g0*np.sqrt(Cf1f1*Cg0g0)
    cov[1,3] = cov[3,1] = Rf0g1*np.sqrt(Cf1f1*Cg1g1)
    cov[2,3] = cov[3,2] = Rg0g1*np.sqrt(Cg0g0*Cg1g1)
    #view_corr(cov,"individual_params")

    ind = 2
    kz = k[ind]
    red = redshifts[ind]
    print "At z = %.2f"%red
    cov_fg = np.zeros((2,2))
    cov_fg[0,0] = cov[0,0]+kz*cov[0,1] + kz**2*cov[1,1] #f,f
    cov_fg[1,1] = cov[2,2]+kz*cov[2,3] + kz**2*cov[3,3] #g,g
    cov_fg[0,1] = cov_fg[1,0] = cov[0,2] + kz*(cov[0,3]+cov[1,2]) + kz**2*cov[1,3]
    #view_corr(cov_fg,"tinker params at z=%.2f"%red)

    dNdfxdNdf = dNdfxdNdf_array[0]
    dNdgxdNdg = dNdgxdNdg_array[0]
    dNdfxdNdg = dNdfxdNdg_array[0]
    cov_model = dNdfxdNdf*cov_fg[0,0] + dNdgxdNdg*cov_fg[1,1] + (dNdfxdNdg + dNdfxdNdg.T)*cov_fg[0,1]
    
    print cov_fg
    print dNdfxdNdf[0,1]*cov_fg[0,0], dNdgxdNdg[0,1]*cov_fg[1,1],(dNdfxdNdg[0,1] + dNdfxdNdg.T[0,1])*cov_fg[0,1]
    print dNdfxdNdf[0,0]*cov_fg[0,0], dNdgxdNdg[0,0]*cov_fg[1,1],(dNdfxdNdg[0,0] + dNdfxdNdg.T[0,0])*cov_fg[0,1]
    print dNdfxdNdf[1,1]*cov_fg[0,0], dNdgxdNdg[1,1]*cov_fg[1,1],(dNdfxdNdg[1,1] + dNdfxdNdg.T[1,1])*cov_fg[0,1]

    print ""
    print np.diag(cov_model)

    def view_corr2(cov,title,x):
        corr = np.zeros_like(cov)
        for i in range(len(cov)):
            for j  in range(len(cov[i])):
                #print cov[i,j],cov[i,i],cov[j,j]
                corr[i,j] = cov[i,j]/np.sqrt(cov[i,i]*cov[j,j])
        plt.pcolor(np.flipud(corr),vmin=-1.0,vmax=1.0)
        ax = plt.gca()
        step = 1
        ticks = np.array([float(i*step) for i in range(len(x)/step)])
        ax.set_xticks(ticks+0.5,minor=False)
        labels = ["%.2f"%x[i*step] for i in range(len(x)/step)]
        ax.set_xticklabels(labels,minor=False)
        ax.set_yticks(ticks+0.5,minor=False)
        labels2=copy.copy(labels)
        labels2.reverse()
        ax.set_yticklabels(labels2,minor=False)
        plt.title(title,fontsize=28)
        plt.colorbar()
        #plt.savefig("./figures/"+title+".png")
        plt.show()
        return corr
    base = "/home/tmcclintock/Desktop/all_MF_data/BACKUP_building_MF_data/"
    datapath = base+"/full_mf_data/Box%03d_full/Box%03d_full_Z%d.txt"
    covpath = base+"/covariances/Box%03d_cov/Box%03d_cov_Z%d.txt"
    MF_data = np.genfromtxt(datapath%(0,0,ind))
    cov_data = np.genfromtxt(covpath%(0,0,ind))
    lM_bins = MF_data[:,:2]
    lM = np.mean(lM_bins,1)
    for i in range(len(cov_data)):
        print "%d\t%.2e\t%.2e\t%.2f"%(i,cov_data[0,i],cov_model[0,i],cov_data[0,i]/cov_model[0,i])
    #print cov_data[0]
    print cov_model[0]
    view_corr2(cov_data,r"$R_{JK}$",lM)
    view_corr2(cov_model,r"$R_{p,p}$",lM)
    view_corr2(cov_data+cov_model,r"$R_{\rm total}$",lM)
