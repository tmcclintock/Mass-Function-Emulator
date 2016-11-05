"""
This contains an actual mass function emulator
that emulates the tinker mass function.
"""

import tinker_mass_function as TMF

class mf_emulator(object):
    default_cosmo_dict = {"om":0.3,"ob":0.05,"ol":1.-0.3,\
                              "ok":0.0,"h":0.7,"s8":0.77,\
                              "ns":3.0,"w0":.96,"wa":0.0}
    """
    This must take a name so that it can save and load.

    This may take a cosmology, a redshift,
    and mass bounds to emulate between.
    """
    def __init__(self,name,cosmo_dict=None,redshift=0.0,l10M_bounds=[11,16]):
        self.name = name
        if cosmo_dict is None:
            self.cosmo_dict = default_cosmo_dict
        self.MF_model = TMF.MF_model(cosmo_dict,redshift,l10M_bounds)
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
    def train(self,cosmologies,training_data):
        """
        Name:
            train
        Purpose:
            Train each of the emulators that handle the incoming parameters
        Input:
            cosmologies (2D array): 
        """
        self.trained = True
        return
    
    def predict_parameters(self,cosmo_dict,redshift):
        """incomplete"""
        #Disassemble the dictionary to get at the parameters
        sf = 1./(1+redshift)
        #Access the emulators and return the predicted parameters
        #along with the variances
        return [0,0,0,0],[1.0,1.0,1.0,1.0]

    def predict_mass_function(self,cosmo_dict,redshift):
        """incomplete"""
        params,variances = self.predict_parameters(cosmo_dict,redshift):
        d,e,f,g = params
        self.MF_model.set_parameters(d,e,f,g)
        self.MF_model.set_new_cosmology(cosmo_dict)

        
