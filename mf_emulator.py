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
    def __init__(self,name,cosmo_dict=None,redshift=0.0,l10M_bounds=[11,16]):
        default_cosmo_dict = {"om":0.3,"ob":0.05,"ol":1.-0.3,\
                                  "ok":0.0,"h":0.7,"s8":0.77,\
                                  "ns":3.0,"w0":.96,"wa":0.0}
        self.name = name
        if cosmo_dict is None:
            self.cosmo_dict = default_cosmo_dict
        #self.MF_model = TMF.MF_model(self.cosmo_dict,redshift,l10M_bounds)
        self.redshift = redshift
        self.scale_factor = 1./(1.+self.redshift)
        self.l10M_bounds = l10M_bounds
        self.trained = False
        self.model_ready = False

    """
    Given an array of cosmologies
    as well as arrays of f0, f1, g0 and g1
    values, train the emulators
    """
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
        N_emulators = len(training_data[0])/2
        emulator_list = []
        print cosmologies.shape, training_data.shape, N_emulators
        for i in range(N_emulators):
            y = training_data[:,i*2]
            yerr = training_data[:,i*2+1]
            emu = emulator.Emulator(name=names[i],xdata=cosmologies,ydata=y,yerr=yerr)
            emu.train()
            emulator_list.append(emu)

        self.emulator_list = emulator_list
        self.trained = True
        return
    
    def predict_parameters(self,cosmology):
        """Docstring TODO"""
        if not self.trained:raise AttributeError("Need to train before predicting")
        #Access the emulators and return the predicted parameters
        out = [emu.predict_one_point(cosmology) for emu in self.emulator_list]
        return np.array(out)

    def predict_mass_function(self,cosmo_dict,redshift):
        """Docstring TODO"""
        params,variances = self.predict_parameters(cosmo_dict,redshift)
        d,e,f,g = params
        self.MF_model.set_parameters(d,e,f,g)
        self.MF_model.set_new_cosmology(cosmo_dict)
        return
        
if __name__=="__main__":
    #Read in the input cosmologies
    all_cosmologies = np.genfromtxt("./cosmology_files/building_cosmos_all_params.txt")
    all_cosmologies = np.delete(all_cosmologies,5,1) #Delete ln10As
    all_cosmologies = np.delete(all_cosmologies,0,1) #Delete boxnum
    all_cosmologies = np.delete(all_cosmologies,-1,0)#39 is broken
    N_cosmologies = len(all_cosmologies)
    print N_cosmologies

    #Read in the input data
    lines = np.loadtxt("training_data/txt_files/lines.txt")
    covs = np.load("training_data/txt_files/line_covs.npy")
    print lines.shape, covs.shape
    N_params = len(lines[0])
    data = np.ones((N_params*2,N_cosmologies))
    for i in range(N_params):
        data[i*2] = lines[:,i]
        data[i*2+1] = np.sqrt(covs[:,i/2,i%2,i%2]) # sorry...
    data = data.T
    print data.shape

    training_cosmologies = all_cosmologies[1:]
    training_data = data[1:]
    test_cosmo = all_cosmologies[0]
    test_data = data[0]
    mf_emulator = mf_emulator("test")
    mf_emulator.train(training_cosmologies,training_data)
    predicted = mf_emulator.predict_parameters(test_cosmo)

    for i in range(len(predicted)):
        
        print "%f %f"%(predicted[i,0],predicted[i][1]),test_data[i*2:i*2+2]
    
