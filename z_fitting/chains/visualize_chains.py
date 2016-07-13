"""
This is a short script that visualizes the f,g paramters for 
"""

import numpy as np
import matplotlib.pyplot as plt
import corner

scale_factors = np.array([0.25,0.333333,0.5,0.540541,0.588235,0.645161,0.714286,0.8,0.909091,1.0])
z = 1./scale_factors - 1.0
sf = scale_factors

path = "mcmc_chains/box%03d_Z%d_chain.txt"

for i in xrange(0,39): #Nfiles
    farr = []
    garr = []
    ferrs = []
    gerrs = []
    for j in xrange(0,10): #Nreds
        data = np.genfromtxt(path%(i,j))
        f,g = np.mean(data,0)
        ferr,gerr = np.std(data,0)
        farr.append(f)
        garr.append(g)
        ferrs.append(ferr)
        gerrs.append(gerr)
        print i,j,z[j],farr[j], garr[j]
        
        fig = corner.corner(data,labels=["f","g"])
        fig.savefig("plots/corner_plots/corner_Box%03d_Z%d.png"%(i,j))
        plt.close()
        
    """np.savetxt("farr.txt",farr)
    np.savetxt("garr.txt",garr)
    np.savetxt("ferr.txt",ferrs)
    np.savetxt("gerr.txt",gerrs)
    np.savetxt("redshifts.txt",z)
    np.savetxt("scale_factors.txt",scale_factors)"""

    """
    #plt.errorbar(sf,farr,ferrs,marker='o',ls='-',ms=2,alpha=0.5)#,label="f")
    #plt.errorbar(sf,garr,gerrs,marker='o',ls='-',ms=2,alpha=0.5)#,label="g")

    leg = plt.legend(loc=0,fancybox=True)
    #leg.get_frame().set_alpha(0.5)
    plt.xlim(-0.1,3.1)
    plt.xlim(0.2,1.1)
    #plt.xlabel("redshift",fontsize=28)
    plt.xlabel("scale_factor",fontsize=28)
    plt.ylabel("Tinker08 parameters",fontsize=28)
    plt.subplots_adjust(bottom=0.15)
    plt.show()"""
        
