import numpy as np

"""
For each cosmology the likelihood is
lnL = -0.5*[ (NM_data-NM_model)_i*(cov_data+cov_model)^-1*(NM_data-NM_model)_j  + log(det|cov_data+cov_model|)|
"""

#Define our lnlikelihood
def lnlike(params,NM_model_array,NM_data_array,cov_data_array,\
               ff_derivs,gg_derivs,fg_derivs,k_array,Ncosmos,Nreds,Zinds):
    cf0f0,cf1f1,cg0g0,cg1g1 = np.exp(params[:4])
    rf0f1,rf0g0,rf0g1,rf1g0,rf1g1,rg0g1 = params[4:]
    #The covariances of the parameter matrix
    cf0f1 = rf0f1*np.sqrt(cf0f0*cf1f1)
    cf0g0 = rf0g0*np.sqrt(cf0f0*cg0g0)
    cf0g1 = rf0g1*np.sqrt(cf0f0*cg1g1)
    cf1g1 = rf1g1*np.sqrt(cf1f1*cg1g1)
    cf1g0 = rf1g0*np.sqrt(cf1f1*cg0g0)
    cg0g1 = rg0g1*np.sqrt(cg0g0*cg1g1)
    lnl_array = np.zeros((Ncosmos))
    #Loop over cosmologies
    for i in xrange(0,Ncosmos):#len(lnl_array),1):
        j = Zinds[i]
        index = i*Nreds+j
        NM_data  = NM_data_array[index]
        NM_model = NM_model_array[index]
        cov_data = cov_data_array[index]
        dNdfxdNdf = ff_derivs[index]
        dNdgxdNdg = gg_derivs[index]
        dNdfxdNdg = fg_derivs[index]
        diff = NM_data - NM_model
        k = k_array[j]
        #Assemble the fg covariance matrix
        cov_ff = cf0f0 + 2*k*cf0f1 + k**2*cf1f1
        cov_gg = cg0g0 + 2*k*cg0g1 + k**2*cg1g1
        cov_fg = cf0f0 + k*cf0g1 + k*cf1g0 + k**2*cf1g1
        #Assemble the NM covariance matrix
        cov_model = dNdfxdNdf*cov_ff + dNdgxdNdg*cov_gg + (dNdfxdNdg + dNdfxdNdg.T)*cov_fg
        #Find the inverse of the full covariance matrix
        cov = cov_data + cov_model
        det = np.linalg.det(cov)
        icov = np.linalg.inv(cov)
        chi2 = np.dot(diff,np.dot(icov,diff))
        if det < 0.0: return -np.inf #Unphysical
        prefactor = np.log(det)
        lnl_array[i] = prefactor+chi2
        continue #end i
    if any(np.isnan(lnl_array).flatten()):
        return -np.inf
    #Return the lnlikelihood
    return -0.5*np.sum(lnl_array)

def lnprior(params):
    if any(np.fabs(params[:4]) > 20): return -np.inf
    if any(np.fabs(params[4:]) > 1):  return -np.inf
    return np.sum(np.exp(params[:4]))

def lnprob(params,NM_model,NM_data,cov_data,ff_derivs,gg_derivs,fg_derivs,k,Ncosmos,Nreds,Zinds):
    lp = lnprior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(params,NM_model,NM_data,cov_data,ff_derivs,gg_derivs,fg_derivs,k,Ncosmos,Nreds,Zinds)
