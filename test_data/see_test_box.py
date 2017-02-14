import numpy as np
import sys
sys.path.insert(0,"../visualization/")
import visualize
import matplotlib.pyplot as plt

Nz = 10
for i in range(0,Nz):
    lMlo,lMhi,N,Nerr = np.genfromtxt("TestBox000_mean_Z%d.txt"%i).T
    lM = np.mean([lMlo,lMhi],0)
    lM = lM[N>0.0]
    Nerr = np.sqrt(N)/5.
    Nerr = Nerr[N>0.0]
    N = N[N>0.0]
    plt.errorbar(lM,N,yerr=Nerr,color=plt.cm.seismic(1.-1.*i/(Nz-1)))
plt.yscale('log')
ylims = plt.gca().get_ylim()
plt.xlabel(r"$\log_{10}M\ [{\rm M_\odot/h}]$")
plt.ylabel(r"${\rm Number}/[1\ {\rm Gpc^3}\log_{10}{\rm M_\odot}]$")
plt.subplots_adjust(bottom=0.15,left=0.15,hspace=0.001)
plt.title("Averaged TestBox000")
plt.show()
plt.close()
