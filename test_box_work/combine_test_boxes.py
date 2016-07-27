"""
This file takes in the test box data and combines the bins
and the covariance matrices.
"""
import numpy as np
import os, sys
sys.path.insert(0,"../visualization/")
import visualize

visualize_curves = False

Ntests = 7
Nreals = 5
Nreds = 10

scale_factors = np.array([0.25,0.333333,0.5,0.540541,0.588235,\
                              0.645161,0.714286,0.8,0.909091,1.0])
redshifts = 1./scale_factors - 1.0
volume = 1.e9 #(1000.)**3 #(Mpc/h)^3

inboxname = "TestBox%03d-%03d"
data_base = "/home/tmcclintock/Desktop/Test_NM_data/"
data_path = data_base+"full_mf_data/%s_full/%s_full_Z%d.txt"
cov_path = data_base+"covariances/%s_cov/%s_cov_Z%d.txt"

outboxname = "TestBox%03d_combined"
out_data_path = data_base+"combined_full_mf_data/%s_full/%s_full_Z%d.txt"
out_cov_path = data_base+"combined_covariances/%s_cov/%s_cov_Z%d.txt"

"""
Read in one sample piece of data so that
we know the shape of the data.
"""
data_sample_path = data_path%(inboxname%(0,0),inboxname%(0,0),0)
cov_sample_path = cov_path%(inboxname%(0,0),inboxname%(0,0),0)
data_sample = np.genfromtxt(data_sample_path)*0
cov_sample = np.genfromtxt(cov_sample_path)*0
print data_sample.shape, cov_sample.shape

"""
Loop over each cosmology
"""
for box_index in xrange(0,Ntests):
    #Define this outboxname
    thisoutbox = outboxname%(box_index)

    print "Combining %s"%thisoutbox
    """
    Loop over redshifts
    """
    for z_index in xrange(0,Nreds):
        print "\tfor Z%d"%z_index
        #First try to make the ouput directory
        os.system("mkdir -p %s"%(data_base+"/combined_full_mf_data/%s_full/"%thisoutbox))
        os.system("mkdir -p %s"%(data_base+"/combined_covariances/%s_cov/"%thisoutbox))

        data_out = np.copy(data_sample)
        cov_out = np.copy(cov_sample)
        
        #Nreals=1
        for r_index in xrange(0,Nreals):
            data_in = np.genfromtxt(data_path%(inboxname%(box_index,r_index),inboxname%(box_index,r_index),z_index))
            cov_in = np.genfromtxt(cov_path%(inboxname%(box_index,r_index),inboxname%(box_index,r_index),z_index))
            data_out += data_in/Nreals
            cov_out += cov_in/Nreals**2
            continue
        
        if visualize_curves:
            lM_bins = np.mean(data_out[:,0:2],1)
            err = np.sqrt(np.diag(cov_out))
            print lM_bins.shape,err.shape,data_out[:,2].shape
            visualize.single_NM_plot(lM_bins,data_out[:,2],err)
        
        #Save the combined data and the combined covariance
        np.savetxt(out_data_path%(thisoutbox,thisoutbox,z_index),data_out)
        np.savetxt(out_cov_path%(thisoutbox,thisoutbox,z_index),cov_out)
        continue

    print "\tcompleted %s"%thisoutbox
    continue # end box_index
