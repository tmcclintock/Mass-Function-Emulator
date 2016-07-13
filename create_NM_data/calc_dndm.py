"""
This file reads from the binned .txt files and finds dn/dM.

In this alpha version, there is only one list file, and the paths aren't what they will be, since
this script will be transported to calvin soon.

Note: the mass of a DM particle is 3.987e10 Msun/h. The minimum number of particles in a halo
is 100. The minimum mass was 3.9877e12 and the maximum was 6.3936e15.

"""

import numpy as np

path = "/media/tom/Data/Emulator_data/" #Laptop path
#path = "/calvin1/tmcclintock/Emulator_data/" #calvin path

#Loop over file numbers
N_files = 1 #40
for i in range(N_files):
    filename = path+"NM_data/NM_box%03d_Z9.txt"%i
    outfilename = path+"dndM_data/dndM_box%03d_Z9.txt"%i
    outfile = open(outfilename,"w")
    outfile.write("#binnum lMmean |dndM| poisson_err_dndM\n")
    outfile.write("#log base 10 used\n")

    vol = 1.e9

    data = np.genfromtxt(filename).T
    bmin_in,bmax_in,NM_in,Mtotal_in = data
    inds = np.where(NM_in > 0)
    bmin,bmax,NM,Mtotal=bmin_in[inds],bmax_in[inds],NM_in[inds],Mtotal_in[inds]
    Mave = Mtotal/NM #Average mass in each bin used
    dM = (Mave[:-1]-Mave[1:])
    M = (Mave[:-1]+Mave[1:])/2.0 #Mass of dndM is the mean of the bins
    lM = np.log10(M)
    dn = (NM[:-1] - NM[1:])/vol
    dndM = dn/dM
    errnM = np.sqrt(NM)/vol
    err_dndm = (errnM[:-1]+errnM[1:])/dM #Poisson error in dndM

    for k in range(len(dndM)):
        outfile.write("%d\t%e\t%e\t%e\n"%(inds[0][k],lM[k],np.fabs(dndM[k]),np.fabs(err_dndm[k])))
    outfile.close()
    print "Box%03d completed"%i
