"""
This file contains the prior and likelihoods for doing the fits to create
training data.
"""
import numpy as np

#Prior
def lnprior(params):
    #d,e,f,g,lnscat2 = params
    if any(params[:4]<0) or any(params[:4]>5): return -np.inf
    if params[4] < -20 or params[4] > 5: return -np.inf
    return 0

#Likelihood
def lnlike(params,lM_bins,N_data,cov_data,icov_data,volume,MF_model):
    d,e,f,g,lnscat2 = params
    MF_model.set_parameters(d,e,f,g)
    N = MF_model.n_in_bins(lM_bins)*volume
    cov_int = np.diag(np.ones_like(N)*np.exp(lnscat2)*N**2)
    cov = cov_data+cov_int
    icov = np.linalg.inv(cov)
    detcov = np.linalg.det(cov)
    X = N_data - N
    LL = -0.5*np.log(2*np.pi*detcov)
    LL+= -0.5*np.dot(X,np.dot(icov,X))
    return LL

#Likelihood without scatter. Can run much faster
def lnlike_no_scatter(params,lM_bins,N_data,cov_data,icov_data,volume,MF_model):
    d,e,f,g,lnscat2 = params
    MF_model.set_parameters(d,e,f,g)
    N = MF_model.n_in_bins(lM_bins)*volume
    X = N_data - N
    return -0.5*np.dot(X,np.dot(icov_data,X))

#Posterior
def lnprob(params,a,lM_bins,N_data,cov_data,icov_data,volume,MF_model):
    f,g = params
    full_params = np.array([1.97,1.0,f,g,-19.0])
    lp = lnprior(full_params)
    if not np.isfinite(lp): return -np.inf
    #return lp + lnlike(full_params,lM_bins,N_data,cov_data,icov_data,volume,MF_model)
    return lp + lnlike_no_scatter(full_params,lM_bins,N_data,cov_data,icov_data,volume,MF_model)
