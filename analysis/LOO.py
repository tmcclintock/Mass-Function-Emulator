import numpy as np
import sys
sys.path.insert(0,"../")
import mf_emulator as MFE

#Scale factors and redshifts and volume
scale_factors = np.array([0.25,0.333333,0.5,0.540541,0.588235,0.645161,0.714286,0.8,0.909091,1.0])
redshifts = 1./scale_factors - 1.0
volume = 1050.**3 #[Mpc/h]^3


#Data path
base = "/home/tmcclintock/Desktop/all_MF_data/building_MF_data/"
datapath = base+"/full_mf_data/Box%03d_full/Box%03d_full_Z%d.txt"
covpath = base+"/covariances/Box%03d_cov/Box%03d_cov_Z%d.txt"

#Number of data files
N_cosmos = 39
N_z = 10

#Read in the cosmologies
all_cosmologies = np.genfromtxt("../test_data/building_cosmos_all_params.txt")
#all_cosmologies = np.delete(all_cosmologies,5,1) #Delete ln10As
all_cosmologies = np.delete(all_cosmologies,0,1) #Delete boxnum
all_cosmologies = np.delete(all_cosmologies,-1,0)#39 is broken

#Read in the input data
means = np.loadtxt("../test_data/mean_models.txt")
variances = np.loadtxt("../test_data/var_models.txt")
data = np.ones((N_cosmos,len(means[0]),2)) #Last column is for mean/erros
data[:,:,0] = means
data[:,:,1] = np.sqrt(variances)

#Create an array with chi2
chi2_array = np.zeros((N_cosmos,N_z))

#Loop over boxes and redshifts
box_low, box_high = 0,N_cosmos
z_low, z_high = 0,N_z
for i in xrange(box_low,box_high):
    test_cosmo = all_cosmologies[i]
    test_data = data[i]
    training_cosmologies = np.delete(all_cosmologies,i,0)
    training_data = np.delete(data,i,0)

    #Create an emulator
    mfe = MFE.mf_emulator("LOO")
    mfe.train(training_cosmologies,training_data)

    #Predict the TMF parameters
    predicted = mfe.predict_parameters(test_cosmo)
    print "\nLOO%d real params: "%i,test_data[:,0]
    print "LOO%d pred params: "%i,predicted[:,0]

    for j in xrange(z_low,z_high):
        MF_data = np.genfromtxt(datapath%(i,i,j))
        lM_bins = MF_data[:,:2]
        N_data = MF_data[:,2]
        cov_data = np.genfromtxt(covpath%(i,i,j))
        icov = np.linalg.inv(cov_data)
        N_err = np.sqrt(np.diagonal(cov_data))
        n = mfe.predict_mass_function(test_cosmo,redshift=redshifts[j],lM_bins=lM_bins)
        N_emu = n*volume
        chi2 = np.dot((N_data-N_emu),np.dot(np.linalg.inv(cov_data),(N_data-N_emu)))
        chi2_array[i,j] = chi2

    print chi2_array[i]

np.savetxt("chi2_array.txt",chi2_array)
