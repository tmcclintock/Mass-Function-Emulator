"""
This contains an actual mass function emulator.
"""
import emulator, sys
import numpy as np
import tinker_mass_function as TMF

#Which data we are working with
dataname = "dfg"

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
        cosmo_dict = {"om":Om,"ob":Ob,"ol":1-Om,"ok":0.0,"h":h,"s8":sigma8,"ns":ns,"w0":w0,"wa":0.0}
        self.cosmology = cosmology
        self.cosmo_dict = cosmo_dict
        self.MF = TMF.tinker_mass_function(cosmo_dict, self.redshift, l10M_bounds)
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
        N_cosmos = len(cosmologies)
        N_emulators = training_data.shape[1]
        emulator_list = []
        for i in range(N_emulators):
            y = training_data[:, i, 0]
            yerr = training_data[:, i, 1]
            emu = emulator.Emulator(name="emu%d"%i, xdata=cosmologies, 
                                    ydata=y, yerr=yerr)
            emu.train()
            emulator_list.append(emu)
        self.emulator_list = emulator_list
        self.trained = True
        return

    def predict_parameters(self,cosmology):
        """Docstring TODO"""
        if not self.trained:raise AttributeError("Need to train before predicting")
        return np.array([emu.predict_one_point(cosmology) for emu in self.emulator_list])

    def predict_mass_function(self, cosmology, redshift, lM_bins, dataname):
        """Docstring TODO"""
        if not self.MF_ready: self.set_cosmology(cosmology,redshift)
        if not all(cosmology==self.cosmology): self.set_cosmology(cosmology,redshift)
        self.redshift = redshift
        a = 1./(1+redshift)
        Tinker_defaults = {'d':1.97, 'e':1.0, "f": 0.51, 'g':1.228}
        def get_params(model, sf):
            if dataname is 'defg':
                d0,d1,e0,e1,f0,f1,g0,g1 = model
            if dataname is 'dfg':
                d0,d1,f0,f1,g0,g1 = model
                e0 = Tinker_defaults['e']
                e1 = 0
            if dataname is 'efg':
                e0,e1,f0,f1,g0,g1 = model
                d0 = Tinker_defaults['d']
                d1 = 0
            if dataname is 'fg':
                f0,f1,g0,g1 = model
                d0 = Tinker_defaults['d']
                d1 = 0
                e0 = Tinker_defaults['e']
                e1 = 0
            k = sf - 0.5
            d = d0 + k*d1
            e = e0 + k*e1
            f = f0 + k*f1
            g = g0 + k*g1
            return d, e, f, g
        predictions = self.predict_parameters(cosmology)
        params, variances = predictions[:,0], predictions[:,1]
        d,e,f,g = get_params(params, a)
        self.MF.set_parameters(d,e,f,g)
        return self.MF.n_in_bins(lM_bins,redshift)
        
if __name__=="__main__":
    #Read in the input cosmologies
    cosmos = np.genfromtxt("./test_data/building_cosmos.txt")
    #cosmos = np.delete(cosmos, [0, 5], 1) #Delete boxnum and ln10As
    cosmos = np.delete(cosmos, 0, 1) #Delete boxnum
    cosmos = np.delete(cosmos, -1, 0)#39 is broken
    N_cosmos = len(cosmos)

    #Read in the input data
    database = "/home/tmcclintock/Desktop/Github_stuff/fit_mass_functions/output/%s/"%dataname
    means     = np.loadtxt(database+"%s_means.txt"%dataname)
    variances = np.loadtxt(database+"%s_vars.txt"%dataname)
    data = np.ones((N_cosmos, len(means[0]),2)) #Last column is for mean/erros
    data[:,:,0] = means
    data[:,:,1] = np.sqrt(variances)
    
    #Pick out the training/testing data
    box, z_index = 0, 1
    test_cosmo = cosmos[box]
    test_data = data[box]
    training_cosmos = np.delete(cosmos, box, 0)
    training_data = np.delete(data, box, 0)

    #Train
    mf_emulator = mf_emulator("test")
    mf_emulator.train(training_cosmos, training_data)

    #Predict the TMF parameters
    predicted = mf_emulator.predict_parameters(test_cosmo)
    print "real params: ",test_data[:,0]
    print "pred params: ",predicted[:,0]

    #Read in the test mass function
    MF_data = np.genfromtxt("./test_data/N_data/Box%03d_full/Box%03d_full_Z%d.txt"%(box, box, z_index))
    lM_bins = MF_data[:,:2]
    N_data = MF_data[:,2]
    cov_data = np.genfromtxt("./test_data/covariances/Box%03d_cov/Box%03d_cov_Z%d.txt"%(box, box, z_index))
    N_err = np.sqrt(np.diagonal(cov_data)) + 0.01*N_data

    #Scale factors and redshifts
    scale_factors = np.array([0.25,0.333333,0.5,0.540541,0.588235,0.645161,0.714286,0.8,0.909091,1.0])
    redshifts = 1./scale_factors - 1.0

    #Predict the TMF
    volume = 1050.**3 #[Mpc/h]^3
    n = mf_emulator.predict_mass_function(test_cosmo,redshift=redshifts[z_index],lM_bins=lM_bins,dataname=dataname)
    N_emu = n*volume
        
    chi2 = np.dot((N_data-N_emu),np.dot(np.linalg.inv(cov_data),(N_data-N_emu)))
    sigdif = (N_data-N_emu)/N_err
    for i in range(len(N_data)):
        print "Bin %d: %.1f +- %.1f\tvs\t%.1f  at  %f"%(i,N_data[i],N_err[i],N_emu[i],sigdif[i])
    print "chi2 = %f"%chi2

    sys.path.insert(0,'./visualization/')
    import visualize
    lM = np.log10(np.mean(10**lM_bins,1))
    visualize.N_comparison(lM, N_data, N_err, N_emu, show=True)


