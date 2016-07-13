"""
This file reads from the single jk files and calculates n(M) with a single region
excluded.

In this alpha version, there is only one list file, and the paths aren't what they will be, since
this script will be transported to calvin soon.
"""

import numpy as np

#path = "/media/tom/Data/Emulator_data/" #Laptop path
path = "/calvin1/tmcclintock/Emulator_data/" #calvin path

#Loop over file numbers
N_files = 40
ndivs = 8
numjks = ndivs**3
for i in range(N_files):

    dataarray = []
    #Loop over all of the input files and assemble them
    for j in range(numjks):
        filename = path+"NM_data/jackknife_singles_NM_data/single_jk%d_NM_box%03d_Z9.txt"%(j,i)
        dataarray.append(np.genfromtxt(filename).T)
    dataarray = np.array(dataarray)
    
    #Create a new combined JK file
    for j in range(numjks):
        outfilename = path+"NM_data/jackknife_NM_data/jk%d_NM_box%03d_Z9.txt"%(j,i)
        outfile = open(outfilename,"w")
        outfile.write("#Binmin Binmax NM Mtotal\n")
        
        #Read in from the single file to get out the edges, mostly
        bmin,bmax,NM,Mtotal = dataarray[j].copy()
        NM*=0
        Mtotal*=0

        #Loop over all the other regions and combine those properly
        for k in range(numjks):
            if j==k:
                continue
            bmink,bmaxk,NMk,Mtotalk = dataarray[k]
            NM+=NMk
            Mtotal+=Mtotalk
        #Print to the output file
        for k in range(len(bmin)):
            outfile.write("%e\t%e\t%e\t%e\n"%(bmin[k],bmax[k],NM[k],Mtotal[k]))
        outfile.close()
    print "Box%03d completed"%i
