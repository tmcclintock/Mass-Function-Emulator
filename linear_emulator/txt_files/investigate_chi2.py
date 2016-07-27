"""
This short program takes all of the LOO chi2 and dof values and
histograms them to see if they match a chi2 distribution.
"""

import numpy as np
import scipy.stats as stats
from scipy.stats import chi2
import matplotlib.pyplot as plt
plt.rc('text',usetex=True, fontsize=20)

scale_factors = np.array([0.25,0.333333,0.5,0.540541,0.588235,0.645161,0.714286,0.8,0.909091,1.0])
redshifts = 1./scale_factors - 1.0

Ncosmos = 39
Nreds = 10

chi2_array = np.genfromtxt("LOO_chi2.txt")
dof_array = np.genfromtxt("LOO_dof.txt")
chi2pd = chi2_array/dof_array

print "Mean chi2 = %f"%np.mean(chi2_array)
print "Mean chi2/dof = %f"%np.mean(chi2_array/dof_array)
print "Mean dof = %f"%np.mean(dof_array)

chi2z_array = np.zeros((Nreds))
varchi2z = np.zeros((Nreds))
for i in xrange(0,Nreds):
    chi2z_array[i] = np.sum(chi2_array[:,i]/dof_array[:,i])/Nreds
    varchi2z[i] = np.var(chi2_array[:,i]/dof_array[:,i])
    print "z = %f\tchi2 = %f"%(redshifts[i],chi2z_array[i])

plt.errorbar(redshifts,chi2z_array,np.sqrt(varchi2z))
plt.xlim(-0.1,3.1)
plt.xlabel("Redshift")
plt.ylabel(r"$\chi^2/{\rm dof}$")
plt.show()

df = np.mean(dof_array)
mean,var,skew,kurt = chi2.stats(df,moments='mvsk')

x = np.linspace(chi2.ppf(0.01,df),chi2.ppf(0.99,df),100)
#plt.plot(x,chi2.pdf(x,df),'r-',lw=5,alpha=0.6)

#plt.hist(chi2_array,normed=True)
plt.hist(chi2pd,normed=True)
plt.xlabel(r"\chi^2/{\rm dof}")
plt.title("Mean dof = %.2f"%df)
plt.subplots_adjust(bottom=0.15)

plt.show()
