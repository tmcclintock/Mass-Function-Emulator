"""
This contains an actual mass function emulator
that emulates the tinker mass function.
"""
import emulator
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
        self.MF_model = TMF.MF_model(cosmo_dict,self.redshift,l10M_bounds)
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
        f0,f1,g0,g1 = params
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
    lines = np.loadtxt("./test_data/lines.txt")
    covs = np.load("./test_data/line_covs.npy")
    data = np.ones((N_cosmologies,len(lines[0]),2))
    for i in range(len(lines[0])):
        data[:,i,0] = lines[:,i]
        data[:,i,1] = np.sqrt(covs[:,i/2,i%2,i%2]) # sorry...
        continue
    
    #Pick out the training data
    training_cosmologies = all_cosmologies[1:]
    training_data = data[1:]
    test_cosmo = all_cosmologies[0]
    test_data = data[0]

    #Train
    mf_emulator = mf_emulator("test")
    mf_emulator.train(training_cosmologies,training_data)

    #Predict the TMF parameters
    predicted = mf_emulator.predict_parameters(test_cosmo)
    f = predicted[0]+0.5*predicted[1]
    g = predicted[2]+0.5*predicted[3]
    print "reals: ",test_data[:,0]
    print "preds: ",predicted[:,0]

    #Read in the test mass function
    MF_data = np.genfromtxt("./test_data/Box000_full_Z9.txt")
    lM_bins = MF_data[:,:2]
    N_data = MF_data[:,2]
    N_data_err = np.sqrt(np.diagonal(np.genfromtxt("./test_data/Box000_cov_Z9.txt")))

    #Predict the TMF
    volume = 1050.**3 #[Mpc/h]^3
    n = mf_emulator.predict_mass_function(test_cosmo,redshift=0.0,lM_bins=lM_bins)
    N_emu = n*volume

    for i in range(len(N_data)):
        print "Bin %d: %.1f +- %.1f\tvs\t%.1f"%(i,N_data[i],N_data_err[i],N_emu[i])



