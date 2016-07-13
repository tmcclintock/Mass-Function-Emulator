"""
This file loops over the sims and figures out the DM particle mass and the minimum/maximum mass
of a halo.
"""

import numpy as np

path = "/media/tom/Data/Emulator_data/"

#Open the facts file
facts = open("facts_file.txt","w")
facts.write("#boxnum Mmin Mmax Mpart\n")

N_files = 1 #40
for i in range(N_files):
    #Read in the files
    filename = path+"reduced_data/reduced_box%03d_Z9.list"%i
    infile = open(filename,"r")    

    Mmin = 1e30
    Mmax = 0
    Npmin = 99999
    Mpart = 0

    for line in infile:
        if line[0]=='#': #skip comments
            continue
        parts = line.split()
        M200b = float(parts[2])
        Np = int(parts[7])
        if M200b > Mmax:
            Mmax = M200b
        if M200b < Mmin and Np < Npmin:
            Mmin = M200b
            Npmin = Np
            Mpart = Mmin/Npmin
    facts.write("%03d\t%e\t%e\t%e\n"%(i,Mmin,Mmax,Mpart))
facts.close()
