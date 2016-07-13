"""
This file reads from the reduced .list files and 
sorts them into reduced jackknife region files.

In this alpha version, there is only one list file, 
and the paths aren't what they will be, since
this script will be transported to calvin soon.

Note: the mass of a DM particle is 3.987e10 Msun/h. 
The minimum number of particles in a halo
was found to be 2 (?!), so reducing was necessary.
"""

import numpy as np

path = "/calvin1/tmcclintock/Emulator_data/" #calvin path

#Number of jackknife regions
ndivs = 8
numjks = ndivs**3

xmin,xmax = 0.0,1050.0001 #the min and max dimensions
step = (xmax - xmin)/ndivs #the spatial step size

#Loop over file numbers
N_files = 40
for i in range(N_files):
    filename = path+"reduced_data/reduced_box%03d_Z9.list"%i
    infile = open(filename,"r")

    #Loop over jackknife regions and open each output file
    outfilearray = []
    for jk in range(numjks):
        
        outfilename = path+"reduced_data/jackknife_files/jk%d_reduced_box%03d_Z9.list"%(jk,i)
        outfilearray.append(open(outfilename,"w"))

    for line in infile:
        if line[0]=='#': #Skip commments
            continue
        parts = line.split()
        x,y,z = parts[8:11]
        x,y,z = float(x),float(y),float(z)
        ix,iy,iz = int(x/step),int(y/step),int(z/step)
        index = iz*ndivs*ndivs + iy*ndivs + ix
        outfilearray[index].write(line)
        continue #end for lines
    
    for jk in range(numjks):
        outfilearray[jk].close()
    print "Box %03d complete"%i
