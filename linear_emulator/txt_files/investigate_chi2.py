"""
This short program takes all of the LOO chi2 and dof values and
histograms them to see if they match a chi2 distribution.
"""

import numpy as np
import scipy.stats as stats
from scipy.stats import chi2
import matplotlib.pyplot as plt
plt.rc('text',usetex=True, fontsize=20)

Ncosmos = 39
Nreds = 10

chi2_array = np.genfromtxt("LOO_chi2.txt")
dof_array = np.genfromtxt("LOO_dof.txt")
chi2pd = chi2_array/dof_array

print "Mean chi2 = %f"%np.mean(chi2_array)
print "Mean chi2/dof = %f"%np.mean(chi2_array/dof_array)
print "Mean dof = %f"%np.mean(dof_array)

df = np.mean(dof_array)
mean,var,skew,kurt = chi2.stats(df,moments='mvsk')

x = np.linspace(chi2.ppf(0.01,df),\
                    chi2.ppf(0.99,df),100)
#plt.plot(x,chi2.pdf(x,df),'r-',lw=5,alpha=0.6)

#plt.hist(chi2_array,normed=True)
plt.hist(chi2pd,normed=True)
plt.xlabel(r"\chi^2/{\rm dof}")
plt.title("Mean dof = %.2f"%df)
plt.subplots_adjust(bottom=0.15)

plt.show()
