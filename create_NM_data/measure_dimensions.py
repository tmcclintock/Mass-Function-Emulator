"""
This is a short script to measure the dimensions of the simulations.

It reads in the reduced .list files and figures out the min and max
positions in each dimension
"""

import numpy as np

#path = "/media/tom/Data/Emulator_data/" #Laptop path
path = "/calvin1/tmcclintock/Emulator_data/" #calvin path

#Open a dimensions file
facts = open("dimensions_file.txt","w")
facts.write("#boxnum xmin xmax ymin ymax zmin zmax\n")

#Loop over file numbers
N_files = 40
for i in range(N_files):
    xmin = ymin = zmin = 10000 #way too big
    xmax = ymax = zmax = -xmin #way too small

    filename = path+"reduced_data/reduced_box%03d_Z9.list"%i
    infile = open(filename,"r")
    for line in infile:
        if line[0]=='#': #Skip commments
            continue
        parts = line.split()
        x,y,z = parts[8:11]
        x,y,z = float(x),float(y),float(z)
        if x < xmin:
            xmin = x
        if x > xmax:
            xmax = x
        if y < ymin:
            ymin = y
        if y > ymax:
            ymax = y
        if z < zmin:
            zmin = z
        if z > zmax:
            zmax = z
    print i,xmin,xmax, ymin, ymax, zmin, zmax

