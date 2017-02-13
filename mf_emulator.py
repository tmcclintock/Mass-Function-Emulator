"""
This contains an actual mass function emulator
that emulates the tinker mass function.
"""
import emulator, sys
import numpy as np
import tinker_mass_function as TMF

class mf_emulator(object):
    """
    This must take a name so that it can save and load.

    This may take a cosmology, a redshift,
    and mass bounds to emulate between.
    """
    def __init__(self,name,redshift=0.0):
        self.name = name
        self.redshift = redshift
        self.scale_factor = 1./(1.+redshift)
        self.trained = False
        self.MF_ready = False

    def set_cosmology(self,cosmology,redshift,l10M_bounds=[11,16]):
        """docstring TODO"""
        if len(cosmology)==7: ombh2,omch2,w0,ns,H0,Neff,sigma8 = cosmology
        else: ombh2,omch2,w0,ns,ln10As,H0,Neff,sigma8 = cosmology
        h = H0/100.
        Ob = ombh2/h**2
        Om = Ob + omch2/h**2
        cosmo_dict = {"om":Om,"ob":Ob,"ol":1-Om,\
                          "ok":0.0,"h":h,"s8":sigma8,\
                          "ns":ns,"w0":w0,"wa":0.0}
        self.cosmology = cosmology
        self.cosmo_dict = cosmo_dict
        self.MF_model = TMF.MF_model(cosmo_dict,self.redshift,l10M_bounds,use_numerical_derivatives=True)
        self.redshift = redshift
        self.scale_factor = 1./(1.+self.redshift)
        self.l10M_bounds = l10M_bounds
        self.MF_ready = True

    def train(self,cosmologies,training_data,names=None):
        """
        Name:
            train
        Purpose:
            Train each of the emulators that handle the incoming parameters
        Input:
            cosmologies (2D array): array N_cosmos long with cosmological
                parameters for each cosmology
            training_data (2D array): array N_cosmos long with 
                mass function parameters for each cosmology to be emulated
            names: names of the emulators, there should be as many as 
                there are emulators to be created
        Output:
            none
        Fields set:
            emulator_list: a list with an emulator for each parameter
            trained: set to True onces training is complete
        """
        if names == None:
            names = ["emu_f0","emu_f1","emu_g0","emu_g1"]
        N_cosmos = len(cosmologies)
        N_emulators = training_data.shape[1]
        emulator_list = []
        for i in range(N_emulators):
            y = training_data[:,i,0]
            yerr = training_data[:,i,1]
            emu = emulator.Emulator(name=names[i],xdata=cosmologies,ydata=y,yerr=yerr)
            emu.train()
            emulator_list.append(emu)
        self.emulator_list = emulator_list
        self.trained = True
        return

    def predict_parameters(self,cosmology):
        """Docstring TODO"""
        if not self.trained:raise AttributeError("Need to train before predicting")
        return np.array([emu.predict_one_point(cosmology) for emu in self.emulator_list])

    def predict_mass_function(self,cosmology,redshift,lM_bins):
        """Docstring TODO"""
        if not self.MF_ready: self.set_cosmology(cosmology,redshift)
        if not all(cosmology==self.cosmology): self.set_cosmology(cosmology,redshift)
        self.redshift = redshift
        self.scale_factor = 1./(1+redshift)
        predictions = self.predict_parameters(cosmology)
        params, variances = predictions[:,0], predictions[:,1]
        f0,f1,g0,g1 =  [ 0.4349943, 0.13878528, 1.18302696, -0.03028451]#the real params for Box000 if I want to test them
        f0,f1,g0,g1 =  params
        d,e = 1.97,1.0
        f = f0 + (self.scale_factor-0.5)*f1
        g = g0 + (self.scale_factor-0.5)*g1
        self.MF_model.set_parameters(d,e,f,g)
        return self.MF_model.n_in_bins(lM_bins,redshift)
        
if __name__=="__main__":
    #Read in the input cosmologies
    all_cosmologies = np.genfromtxt("./test_data/building_cosmos_all_params.txt")
    #all_cosmologies = np.delete(all_cosmologies,5,1) #Delete ln10As
    all_cosmologies = np.delete(all_cosmologies,0,1) #Delete boxnum
    all_cosmologies = np.delete(all_cosmologies,-1,0)#39 is broken
    N_cosmologies = len(all_cosmologies)

    #Read in the input data
    means = np.loadtxt("./test_data/mean_models.txt")
    variances = np.loadtxt("./test_data/var_models.txt")
    data = np.ones((N_cosmologies,len(means[0]),2)) #Last column is for mean/erros
    data[:,:,0] = means
    data[:,:,1] = np.sqrt(variances)
    
    #Pick out the training/testing data
    box_index, z_index = 0, 9
    test_cosmo = all_cosmologies[box_index]
    test_data = data[box_index]
    training_cosmologies = np.delete(all_cosmologies,box_index,0)
    training_data = np.delete(data,box_index,0)

    #Train
    mf_emulator = mf_emulator("test")
    mf_emulator.train(training_cosmologies,training_data)

    #Predict the TMF parameters
    predicted = mf_emulator.predict_parameters(test_cosmo)
    print "real params: ",test_data[:,0]
    print "pred params: ",predicted[:,0]

    #Read in the test mass function
    MF_data = np.genfromtxt("./test_data/Box%03d_full_Z%d.txt"%(box_index,z_index))
    MF_data = np.genfromtxt("../../all_MF_data/building_MF_data/full_mf_data/Box000_full/Box%03d_full_Z%d.txt"%(box_index,z_index))
    lM_bins = MF_data[:,:2]
    N_data = MF_data[:,2]
    cov_data = np.genfromtxt("./test_data/Box%03d_cov_Z%d.txt"%(box_index,z_index))
    cov_data = np.genfromtxt("../../all_MF_data/building_MF_data/covariances/Box000_cov/Box%03d_cov_Z%d.txt"%(box_index,z_index))
    N_err = np.sqrt(np.diagonal(cov_data))

    #Scale factors and redshifts
    scale_factors = np.array([0.25,0.333333,0.5,0.540541,0.588235,0.645161,0.714286,0.8,0.909091,1.0])
    redshifts = 1./scale_factors - 1.0

    #Predict the TMF
    volume = 1050.**3 #[Mpc/h]^3
    n = mf_emulator.predict_mass_function(test_cosmo,redshift=redshifts[z_index],lM_bins=lM_bins)
    N_emu = n*volume
        
    chi2 = np.dot((N_data-N_emu),np.dot(np.linalg.inv(cov_data),(N_data-N_emu)))
    sigdif = (N_data-N_emu)/N_err
    for i in range(len(N_data)):
        print "Bin %d: %.1f +- %.1f\tvs\t%.1f  at  %f"%(i,N_data[i],N_err[i],N_emu[i],sigdif[i])
    print "chi2 = %f"%chi2

    sys.path.insert(0,'./visualization/')
    import visualize
    lM = np.log10(np.mean(10**lM_bins,1))
    #visualize.NM_plot(lM,N_data,N_err,lM,N_emu,title=r"LOO Box%03d at z=%.2f $\chi^2=%.2f$"%(box_index,redshifts[z_index],chi2))
    visualize.NM_plot(lM,N_data,N_err,lM,N_emu)


