"""
This file fits the mass function data from each of the sims and writes the 
best fit parameters to a file called best_params.txt. The format of that
file is:
#Box d e f g
XXX f f f f
...
"""

import numpy as np
import scipy.optimize as op
import emcee
import sys
sys.path.insert(0,'NM_model')
import NM_model as NM_model_module
import matplotlib.pyplot as plt
import visualize


scale_factors = np.array([0.25,0.333333,0.5,0.540541,0.588235,0.645161,0.714286,0.8,0.909091,1.0])
redshifts = 1./scale_factors - 1.0

do_test = False
do_best_fit = False
do_bf_comparison = False
do_sigma_plots = False
do_mcmc = False
do_mean_comparison = True

cosmos = np.genfromtxt("cosmos.txt")

volume = 1.e9 #(1000.)**3 #(Mpc/h)^3

data_path = "/home/tmcclintock/Desktop/Mass_Function_Data/full_mf_data/Box%03d_full/Box%03d_full_Z%d.txt"
cov_path = "/home/tmcclintock/Desktop/Mass_Function_Data/covariances/Box%03d_cov/Box%03d_cov_Z%d.txt"

#First write the likelihood functions
#This is the prior
def lnprior(params):
    d,e,f,g,lnsig2_int = params
    if d < 0 or e < 0 or f < 0 or g < 0 or lnsig2_int < -15:
        return -np.inf
    if d > 10 or e > 10 or f > 10 or g > 10 or lnsig2_int > 5:
        return -np.inf
    return 0

#This is the likelihood
def lnlike(params,lM_bins,NM_data,cov_in,NM_model_obj,redshift):
    f,g = params#############
    params = [1.97,1.0,f,g,-10]######
    params_in = [1.97,1.0,f,g]
    cov = cov_in.copy()
    prior = lnprior(params)
    if prior < -1e99: return prior    
    NM_model = NM_model_obj.MF_model_all_bins(lM_bins,params_in,redshift)
    d,e,f,g,ln_scatter = params
    for i in range(len(NM_data)):
        cov[i,i]+=np.exp(ln_scatter)*NM_model[i]**2
    icov = np.linalg.inv(cov)
    detcov = np.linalg.det(cov)
    X = NM_data - NM_model
    LL = 0
    for i in range(len(NM_data)):
        for j in range(len(NM_data)):
            LL += -0.5*X[i]*icov[i,j]*X[j]
    LL+=-0.5*np.log(2*np.pi*detcov)
    return LL

for index in xrange(0,len(cosmos)):
    if index < 0 or index > 2: continue

    #Get in the cosmology
    num,ombh2,omch2,w0,ns,ln10As,H0,Neff,sigma8 = cosmos[index,:]
    h = H0/100.
    Ob,Om = ombh2/(h**2), ombh2/(h**2)+omch2/(h**2)
    cosmo_dict = {"om":Om,"ob":Ob,"ol":1-Om,\
                  "ok":0.0,"h":h,"s8":sigma8,\
                  "ns":ns,"w0":w0,"wa":0.0}

    #Loop over redshifts
    for z_index in xrange(9,10):
        redshift = redshifts[z_index]

        #Read in the NM data
        indata = np.loadtxt(data_path%(index,index,z_index))
        cov = np.loadtxt(cov_path%(index,index,z_index))
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

        #Create the model object
        NM_model_obj = NM_model_module.MF_model(cosmo_dict,bounds,volume,redshift)

        #Starting point for parameters
        guesses = np.array([1.97,1.0,0.51,1.228,1.0]) #d,e,f,g, ln_scatter
        guesses = np.array([0.51,1.228]) #f,g

        #Try just a test call
        if do_test:
            lnlike(guesses,lM_bins,NM_data,cov,NM_model_obj,redshift)

        #Attempt a best fit
        if do_best_fit:
            nll = lambda *args: -lnlike(*args)
            result = op.minimize(nll, guesses,args=(lM_bins,NM_data,cov,NM_model_obj,redshift),method="Powell")#,options={"maxiter":1})
            np.savetxt("chains/best_fits/box%03d_Z%d_best.txt"%(index,z_index),result['x'])
            print "Best fit for Box%03d Z%d:"%(index,z_index)
            print result
            print ""

        if do_bf_comparison:
            best_model = np.loadtxt("chains/best_fits/box%03d_Z%d_best.txt"%(index,z_index))
            f,g = best_model
            best_model = [1.97,1.0,f,g]
            best_NM = NM_model_obj.MF_model_all_bins(lM_bins,best_model,redshift)
            lM = np.log10(np.mean(10**lM_bins,1))
            NM_err = np.sqrt(np.diagonal(cov))
            visualize.NM_plot(lM,NM_data,NM_err,lM,best_NM)
            """import matplotlib.pyplot as plt
            plt.errorbar(lM,NM_data,yerr=np.sqrt(np.diagonal(cov)))
            plt.plot(lM,best_NM)
            plt.yscale('log')
            plt.show()"""
            #comparison.compare(index,best_model)

        if do_sigma_plots:
            best_model = np.loadtxt("chains/best_fits/box%03d_Z%d_best.txt"%(index,z_index))
            f,g = best_model
            best_model = [1.97,1.0,f,g]

            lM = np.log10(np.mean(10**lM_bins,1))
            NM_err = np.sqrt(np.diagonal(cov))
            visualize.g_sigma_plot(NM_model_obj,redshift,volume,cosmo_dict,lM,lM_bins,NM_data,NM_err,best_model)

        if do_mcmc:
            best_model = np.loadtxt("chains/best_fits/box%03d_Z%d_best.txt"%(index,z_index))
            ndim, nwalkers,nsteps,nburn = 2, 6, 5000,1000
            print "Performing MCMC for box%03d Z%d"%(index,z_index)
            pos = [best_model + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
            sampler = emcee.EnsembleSampler(nwalkers,ndim,lnlike,args=(lM_bins,NM_data,cov,NM_model_obj,redshift))
            sampler.run_mcmc(pos,nsteps)
            print "MCMC complete for box%03d"%index
            samples = sampler.chain[:,:,:].reshape((-1,ndim))
            samples = sampler.flatchain[nwalkers*nburn:,:].reshape((-1,ndim))
            likes = sampler.flatlnprobability[nwalkers*nburn:]
            np.savetxt("chains/mcmc_chains/box%03d_Z%d_chain.txt"%(index,z_index),samples)
            np.savetxt("chains/mcmc_chains/box%03d_Z%d_likes.txt"%(index,z_index),likes)
            print "Chain saved for box%03d Z%d\n"%(index,z_index)

        if do_mean_comparison:
            chain = np.genfromtxt("chains/mcmc_chains/box%03d_Z%d_chain.txt"%(index,z_index))
            f,g = np.mean(chain,0)
            print f,g
            #best_model = np.loadtxt("chains/best_fits/box%03d_Z%d_best.txt"%(index,z_index))
            #f,g = best_model
            best_model = [1.97,1.0,f,g]
            best_NM = NM_model_obj.MF_model_all_bins(lM_bins,best_model,redshift)
            lM = np.log10(np.mean(10**lM_bins,1))
            NM_err = np.sqrt(np.diagonal(cov))
            print "Plotting mean fit of Box%03d Z%d"%(index,z_index)
            #visualize.NM_plot(lM,NM_data,NM_err,lM,best_NM,"Box%03d Z%d"%(index,z_index),"plots/mean_fit_plots/mean_fit_Box%03d_Z%d.png"%(index,z_index))
