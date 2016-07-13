"""
This file reads from the reduced .list files and bins the halos logarithmically

In this alpha version, there is only one list file, and the paths aren't what they will be, since
this script will be transported to calvin soon.

This calculated N(M) for single(!!) jackknife regions. A seperate file
combines the single N(M)s calculated into versions with single jkregions removed.
"""

import numpy as np

#path = "/media/tom/Data/Emulator_data/" #Laptop path
path = "/calvin1/tmcclintock/Emulator_data/" #calvin path

#Open the facts file
facts = np.genfromtxt("facts_file.txt")

#Loop over file numbers
N_files = 40
ndivs = 8 #number of jk divs
numjks = ndivs**3
for i in range(N_files):
    #Set up the bin edges based in the minimum and maximum masses
    Nedges = 21
    box, Mmin, Mmax, Mpart = facts[i]
    lMmin, lMmax = np.log10(Mmin), np.log10(Mmax)    
    edges = np.linspace(lMmin,lMmax,21)

    for j in range(numjks):
        #Set up the bins
        NM = np.zeros(Nedges-1)
        Mtotal = np.zeros_like(NM) #Used to find the mean masses

        filename = path+"reduced_data/jackknife_files/jk%d_reduced_box%03d_Z9.list"%(j,i)
        infile = open(filename,"r")   
        outfilename = path+"NM_data/jackknife_singles_NM_data/single_jk%d_NM_box%03d_Z9.txt"%(j,i)
        outfile = open(outfilename,"w")
        outfile.write("#Binmin Binmax N(M) Mtotal\n")
        
        for line in infile:
            if line[0]=='#': #Skip comments
                continue
            parts = line.split()
            M200b = float(parts[2])
            lM200b = np.log10(M200b)
            for k in range(Nedges-1):
                if lM200b >= edges[k] and lM200b < edges[k+1]:
                    Mtotal[k] += M200b
                    NM[k]+=1
                    break
        infile.close()
        for k in range(Nedges-1):
            outfile.write("%e\t%e\t%e\t%e\n"%(edges[k],edges[k+1],NM[k],Mtotal[k]))
        outfile.close()
    print "Box%03d completed"%i
