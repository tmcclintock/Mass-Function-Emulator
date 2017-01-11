"""
We have the covariance parameters available at this stage.
That means that we want to know the distribution of the chi2s.
"""
import numpy as np
import pickle, sys, os, copy

N_cosmos = 39#Number of data files
N_z = 10#Number of redshifts

"""
Load in all the pickled files
Each of these are N_cosmos X N_z = 390 length
arrays that contain objects that are the information
in that array. For instance N_data_array[0]
contains the N_data information for the first
cosmology at the first redshift.
"""
N_data_array = pickle.load(open("N_data_array.p","rb"))
cov_data_array = pickle.load(open("cov_data_array.p","rb"))
N_emu_array = pickle.load(open("N_emu_array.p","rb"))
dNdf_array = pickle.load(open("dNdf_array.p","rb"))
dNdg_array = pickle.load(open("dNdg_array.p","rb"))
dNdfxdNdf_array = pickle.load(open("dNdfxdNdf_array.p","rb"))
dNdgxdNdg_array = pickle.load(open("dNdgxdNdg_array.p","rb"))
dNdfxdNdg_array = pickle.load(open("dNdfxdNdg_array.p","rb"))

#Scale factors and redshifts
scale_factors = np.array([0.25,0.333333,0.5,0.540541,0.588235,
                          0.645161,0.714286,0.8,0.909091,1.0])
redshifts = 1./scale_factors - 1.0
k_all = scale_factors - 0.5 # this is because f = f0 + k*f1

#Get in the covariance parameters and build the hyperparam cov matrix
all_chain_means = np.loadtxt("output_files/chain_means.txt")
fullmeans = np.mean(all_chain_means,0)
Cf0f0,Cf1f1,Cg0g0,Cg1g1 = np.exp(fullmeans[:4])
Rf0f1,Rf0g0,Rf0g1,Rf1g0,Rf1g1,Rg0g1 = fullmeans[4:]
cov_HP = np.diag(np.exp(fullmeans[:4]))
cov_HP[0,1] = cov_HP[1,0] = Rf0f1*np.sqrt(Cf0f0*Cf1f1)
cov_HP[0,2] = cov_HP[2,0] = Rf0g0*np.sqrt(Cf0f0*Cg0g0)
cov_HP[0,3] = cov_HP[3,0] = Rf0g1*np.sqrt(Cf0f0*Cg1g1)
cov_HP[1,2] = cov_HP[2,1] = Rf0g0*np.sqrt(Cf1f1*Cg0g0)
cov_HP[1,3] = cov_HP[3,1] = Rf0g1*np.sqrt(Cf1f1*Cg1g1)
cov_HP[2,3] = cov_HP[3,2] = Rg0g1*np.sqrt(Cg0g0*Cg1g1)

def get_cov_fg(cov,k): #cov is cov_HP outside this function
    cov_fg = np.zeros((2,2))
    cov_fg[0,0] = cov[0,0]+k*cov[0,1] + k**2*cov[1,1] #f,f
    cov_fg[1,1] = cov[2,2]+k*cov[2,3] + k**2*cov[3,3] #g,g
    cov_fg[0,1] = cov_fg[1,0] = cov[0,2]+k*(cov[0,3]+cov[1,2])+k**2*cov[1,3]#f,g
    return cov_fg

def get_cov_model(cov_fg,NfNf,NgNg,NfNg):
    """
    Nf is actually dNdf, and NfNf is dNdfxdNdf
    Same for the others
    """
    cov_model = NfNf*cov_fg[0,0]+NgNg*cov_fg[1,1]+(NfNg+NfNg.T)*cov_fg[0,1]
    return cov_model

#Loop over boxes and redshifts and get all final chi2
box_low, box_high = 0,N_cosmos
z_low, z_high = 0,N_z
Nboxes = box_high - box_low
Nzs = z_high - z_low
chi2s = np.zeros((Nboxes*Nzs))
N_fp = np.zeros((Nboxes*Nzs)) #Number of free parameters
ci = 0 #counter
for i in xrange(box_low,box_high):
    cj = 0 #counter
    for j in xrange(z_low,z_high):
        index = i*N_z + j
        N_data    = N_data_array[index]
        N_emu     = N_emu_array[index]
        cov_data  = cov_data_array[index]
        dNdfxdNdf = dNdfxdNdf_array[index]
        dNdgxdNdg = dNdgxdNdg_array[index]
        dNdfxdNdg = dNdfxdNdg_array[index]
        k = k_all[j]
        cov_fg = get_cov_fg(cov_HP,k)
        cov_model = get_cov_model(cov_fg,dNdfxdNdf,dNdgxdNdg,dNdfxdNdg)
        icov = np.linalg.inv(cov_data+cov_model)
        X = N_data - N_emu
        chi2 = 0.0
        lo,hi = 0, len(X)
        for ii in xrange(lo,hi):
            for jj in xrange(lo,hi):
                chi2 += X[ii]*icov[ii,jj]*X[jj]
        #chi2s[ci*Nzs + cj] = np.dot(X,np.dot(icov,X))
        chi2s[ci*Nzs + cj] = chi2
        N_fp[ci*Nzs + cj] = hi-lo
        cj += 1
        continue
    ci += 1
    continue

import matplotlib.pyplot as plt
from scipy.stats import chi2
plt.hist(chi2s/2.,40,normed=True) #Make the histogram
df = np.mean(N_fp)
mean,var,skew,kurt = chi2.stats(df,moments='mvsk')
x = np.linspace(chi2.ppf(0.01,df),chi2.ppf(0.99,df),100)
plt.plot(x,chi2.pdf(x,df))
plt.show()
