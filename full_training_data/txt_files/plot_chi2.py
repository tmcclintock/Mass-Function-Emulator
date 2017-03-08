"""
Plots the chi2 values calculated in create_training_data.py.
"""
import numpy as np
import os,sys
import matplotlib.pyplot as plt
plt.rc("text",usetex=True,fontsize=24)

N_cosmos = 39#Number of data files
N_z = 10#Number of redshifts

data = np.loadtxt("full_chi2_models.txt")
print data.shape
chi2s = data.flatten()

import matplotlib.pyplot as plt
from scipy.stats import chi2
plt.hist(chi2s,30,normed=True) #Make the histogram
df = 10 #approximately
mean,var,skew,kurt = chi2.stats(df,moments='mvsk')
x = np.linspace(chi2.ppf(0.01,df),chi2.ppf(0.999,df),100)
plt.plot(x,chi2.pdf(x,df))
plt.xlabel(r"$\chi^2$",fontsize=24)
plt.subplots_adjust(bottom=0.15)
plt.show()
