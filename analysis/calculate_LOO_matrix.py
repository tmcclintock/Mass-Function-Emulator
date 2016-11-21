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
import pickle, sys, os
import likelihoods_for_covariance as lhfuncs

#Flow control
do_single_test = True
do_maximization = False
do_MCMC = False

#MCMC information
N_trials = 1#00
N_dim = 6
N_walkers = 3*N_dim #2*N_dim minimum
N_steps = 1000

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
initial_parameter_guess = np.array((-7,-7,-7,-7,0,0,0,0,0,0))

#Perform various actions
lnprob = lhfuncs.lnprob
if do_single_test:
    z_indices = np.random.randint(0,N_z,N_cosmos)
    print lnprob(initial_parameter_guess,N_emu_array,\
                     N_data_array,cov_data_array,\
                     dNdfxdNdf_array,dNdgxdNdg_array,\
                     dNdfxdNdg_array,k,N_cosmos,N_z,z_indices)

if do_maximization:
    z_indices = np.random.randint(0,N_z,N_cosmos)
    nll = lambda *args:-lnprob(*args)
    result = op.minimize(nll,initial_parameter_guess,\
                             args=(N_emu_array,\
                                       N_data_array,cov_data_array,\
                                       dNdfxdNdf_array,dNdgxdNdg_array,\
                                       dNdfxdNdg_array,k,\
                                       N_cosmos,N_z,z_indices),\
                             method="Powell")
    print result
    np.savetxt("output_files/maximization_result.txt",result['x'])

if do_MCMC:
    start = np.loadtxt("output_files/maximization_result.txt")
    for trial in xrange(0,N_trials):
        pos = np.zeros((N_walkers,N_dim))
        for i in range(N_walkers):
            pos[i] += start
            pos[i,:4] += np.ranodm.randn(4)*1e-1
            pos[i,4:] += np.random.randn(6)*1e-1
        
        z_indices = np.random.randint(0,N_z,N_cosmos)
        sampler = emcee.EnsembleSampler(N_walkers,N_dim,lnprob,\
                                            args=(N_emu_array,\
                                                      N_data_array,cov_data_array,\
                                                      dNdfxdNdf_array,dNdgxdNdg_array,\
                                                      dNdfxdNdg_array,k,\
                                                      N_cosmos,N_z,z_indices))
        sampler.run_mcmc(pos,N_steps)
        chain = sampler.chain
        likes = sampler.lnprobability
        np.savetxt("output_files/chains/chain_trail%d.txt"%trial,chain)
        np.savetxt("output_files/chains/likes_trail%d.txt"%trial,likes)
