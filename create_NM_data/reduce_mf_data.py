"""
This file reads from the .list files and trims off halos that have Np < 100.

In this alpha version, there is only one list file, and the paths aren't what they will be, since
this script will be transported to calvin soon.

Note: the mass of a DM particle is 3.987e10 Msun/h. The minimum number of particles in a halo
was found to be 2 (?!), so reducing was necessary.
"""

import numpy as np

#path = "/media/tom/Data/Emulator_data/" #Laptop path
path = "/calvin1/tmcclintock/Emulator_data/" #calvin path

#Open a facts file
facts = open("facts_file.txt","w")
facts.write("#boxnum Mmin Mmax Mpart\n")

#Loop over file numbers
N_files = 40
for i in range(N_files):
    #filename = path+"raw_data/box%03d_Z9.list"%i #Laptop path
    filename = path + "Box_halos_Z9/box%03d_Z9.list"%i #calvin path
    infile = open(filename,"r")
    
    outfilename = path+"reduced_data/reduced_box%03d_Z9.list"%i
    outfile = open(outfilename,"w")
    #First I am looping through to find the minimum/maximum, average, and number of halos
    Nhalos = 0
    Mtotal = 0
    Mmin = 1e30
    Mmax = 0
    Npmin, Npmax = 999999,0
    Mpart = 3.987662e10 #Mass of a single DM particle
    for line in infile:
        if line[0]=='#': #Skip commments
            outfile.write(line)
            continue
        parts = line.split()
        PID = float(parts[-1])
        xmax = 0
        xmin = 1000
        if PID == -1:
            M200b = float(parts[2])
            Np = int(parts[7])
            X = float(parts[8])
            if Np > 199: #Then it's a real halo
                outfile.write(line)
                Nhalos += 1
                if M200b > Mmax:
                    Mmax = M200b
                if M200b < Mmin:
                    Mmin = M200b
                if X > xmax:
                    xmax = X
                if X < xmin:
                    xmin = X
                if Np < Npmin:
                    Npmin = Np
                    Mpart = M200b/Np
                if Np > Npmax:
                    Npmax = Np
                Mtotal += M200b
        continue #end for lines
    facts.write("%03d\t%e\t%e\t%e\n"%(i,Mmin,Mmax,Mpart))
    Mave = Mtotal/Nhalos
    infile.close()
    print "Box%03d"%i
    print "Nhalos = %d"%Nhalos
    print "Mmax = %e\tMmin = %e"%(Mmax,Mmin)
    print "Mtotal = %e\tMave = %e"%(Mtotal,Mave)
    print "Npmax = %d\tNpmin = %d"%(Npmax,Npmin)#(Mmax/Mpart,Mmin/Mpart)
    print "Mpart = %e"%Mpart
    print "Xmax = %f\tXmin = %f\n"%(xmax,xmin)
    
