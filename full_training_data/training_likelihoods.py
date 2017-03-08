"""
This file contains the prior and likelihoods 
for doing the fits to create training data.
"""
import numpy as np

#Likelihood without scatter. Can run much faster
def lnlike_no_scatter(d,e,f,g,a,z,lM_bins,N_data,cov_data,icov_data,volume,MF_model):
    Len = len(a)
    LL = 0
    for i in range(Len):
        MF_model[i].set_parameters(d[i],e[i],f[i],g[i])
        N = MF_model[i].n_in_bins(lM_bins[i])*volume
        X = N_data[i] - N
        LL+= -0.5*np.dot(X,np.dot(icov_data[i],X))
    return LL

#Posterior
def lnprob(params,a,z,lM_bins,N_data,cov_data,icov_data,volume,MF_model):
    d0,d1,f0,f1,g0,g1 = params
    d = d0+(a-0.5)*d1
    e = 1.0*np.ones_like(a)#e0+(a-0.5)*e1 #e
    f = f0+(a-0.5)*f1
    g = g0+(a-0.5)*g1
    if any(d<0) or any(d>5): return -np.inf
    if any(e<0) or any(e>5): return -np.inf
    if any(f<0) or any(f>5): return -np.inf
    if any(g<0) or any(g>5): return -np.inf
    return lnlike_no_scatter(d,e,f,g,a,z,lM_bins,N_data,cov_data,icov_data,volume,MF_model)
