import cosmocalc as cc
from scipy import special
from scipy.special import gamma,digamma
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
import numpy as np

TOL = 1e-3
DELTA = 1e-6

class MF_model(object):
    """
    This function takes in low and high log_10 mass
    values and creates bins (i.e. an Nbinsx2 array 
    where Nbins is the number of bins).
    """
    def make_bins(self,Nbins,lM_low,lM_high):
        lM_edges = np.linspace(lM_low,lM_high,Nbins+1)
        return np.array(zip(lM_edges[:-1],lM_edges[1:]))

    """
    Initialization function. It requires a dictionary
    that contains the cosmology to be passed to
    cosmocalc, the upper and lower bounds of the 
    mass in log_10 space, the
    volume we are calculating the mass function in,
    and the redshift we are working at.
    """
    def __init__(self,cosmo_dict,lM_bounds,volume,redshift):
        self.lM_bounds = lM_bounds
        self.volume = volume #volume of the simulation
        self.redshift = redshift
        self.scale_factor = 1./(1.+self.redshift)
        self.set_new_cosmology(cosmo_dict)

    def set_new_cosmology(self,cosmo_dict):
        #Constants
        G = 4.52e-48 #Newton's gravitional constant in Mpc^3/s^2/Solar Mass
        Mpcperkm = 3.241e-20 #Mpc/km; used to convert H0 to s^-1
        self.cosmo_dict = cosmo_dict
        cc.set_cosmology(cosmo_dict)
        self.build_splines()
        Om,H0 = cosmo_dict["om"],cosmo_dict["h"]*100.0
        self.rhom=Om*3.*(H0*Mpcperkm)**2/(8*np.pi*G*(H0/100.)**2)#Msunh^2/Mpc^3

    def build_splines(self):
        lM_min,lM_max = self.lM_bounds
        M_space = np.logspace(lM_min-1,lM_max+1,500,base=10)
        sigmaM = np.array([cc.sigmaMtophat_exact(M,self.scale_factor) for M in M_space])
        ln_sig_inv_spline = IUS(M_space,-np.log(sigmaM))
        deriv_spline = ln_sig_inv_spline.derivative()
        self.deriv_spline = deriv_spline
        self.splines_built = True
        return

    def B_coeff(self,d,e,f,g):
        return 2.0/(e**d * g**(-d/2.)*gamma(d/2.) + g**(-f/2.)*gamma(f/2.))

    def dB_df(self,d,e,f,g):
        B = self.B_coeff(d,e,f,g)
        return B*B/4.*g**(-f/2.)*gamma(f/2.)*(np.log(g) - digamma(f/2.))

    def dB_dg(self,d,e,f,g):
        B = self.B_coeff(d,e,f,g)
        return B*B/4.*(d*e**d*g**(-d/2.)/g*gamma(d/2.) + f*g**(-f/2.)/g*gamma(f/2.))

    def dg_df(self,sigma,params):
        d,e,f,g = params
        dBdf = self.dB_df(d,e,f,g)
        B = self.B_coeff(d,e,f,g)
        return dBdf*((sigma/e)**-d+sigma**-f)*np.exp(-g/sigma**2) - B*sigma**-f*np.log(sigma)*np.exp(-g/sigma**2)

    def dg_dg(self,sigma,params):
        d,e,f,g = params
        dBdg = self.dB_dg(d,e,f,g)
        g_sigma = self.calc_g(sigma,params)
        return dBdg*np.exp(-g/sigma**2)*((sigma/e)**-d + sigma**-f) - g_sigma/sigma**2
        
    def calc_g(self,sigma,params):
        d,e,f,g = params
        return self.B_coeff(d,e,f,g)*((sigma/e)**-d + sigma**-f) * np.exp(-g/sigma**2)

    def ddf_dndM_at_M(self,lM,params):
        rhom,dln_sig_inv_dM_spline = self.rhom,self.deriv_spline
        M = np.exp(lM)
        dgdf = self.dg_df(cc.sigmaMtophat(M,self.scale_factor),params)
        return dgdf*rhom*dln_sig_inv_dM_spline(M) #*M/M #log integral

    def ddg_dndM_at_M(self,lM,params):
        rhom,dln_sig_inv_dM_spline = self.rhom,self.deriv_spline
        M = np.exp(lM)
        dgdg = self.dg_dg(cc.sigmaMtophat(M,self.scale_factor),params)
        return dgdg*rhom*dln_sig_inv_dM_spline(M) #*M/M #log integral

    def dndM_at_M(self,lM,params):
        rhom,dln_sig_inv_dM_spline = self.rhom,self.deriv_spline
        M = np.exp(lM)
        g_sigma = self.calc_g(cc.sigmaMtophat(M,self.scale_factor),params)
        return g_sigma * rhom*dln_sig_inv_dM_spline(M) #*M/M #log integral
    
    def var_MF_model_in_bin(self,lMlow,lMhigh,params,variances,cov_fg = 0):
        dNdf = integrate.quad(self.ddf_dndM_at_M,lMlow,lMhigh,args=(params))[0]*self.volume
        dNdg = integrate.quad(self.ddg_dndM_at_M,lMlow,lMhigh,args=(params))[0]*self.volume
        
        #fparams1 = np.copy(params)
        #df = params[2]*DELTA
        #fparams1[2] += df/2.
        #fparams2 = np.copy(params)
        #fparams2[2] -= df/2.
        #dNdf = (integrate.quad(self.dndM_at_M,lMlow,lMhigh,args=(fparams1))[0] - integrate.quad(self.dndM_at_M,lMlow,lMhigh,args=(fparams2))[0])/df*self.volume

        #gparams1 = np.copy(params)
        #dg = params[3]*DELTA
        #gparams1[3] += dg/2.
        #gparams2 = np.copy(params)
        #gparams2[3] -= dg/2.
        #dNdg = (integrate.quad(self.dndM_at_M,lMlow,lMhigh,args=(gparams1))[0] - integrate.quad(self.dndM_at_M,lMlow,lMhigh,args=(gparams2))[0])/dg*self.volume

        #print ""
        #print dNdf, variances[0], dNdf**2*variances[0]
        #print dNdg, variances[1], dNdg**2*variances[1]
        #print dNdf*dNdg, cov_fg, 2*dNdf*dNdg*cov_fg
        #print dNdf**2*variances[0] + dNdg**2*variances[1] + 2*dNdf*dNdg*cov_fg
        
        return dNdf**2*variances[0] + dNdg**2*variances[1] + 2*dNdf*dNdg*cov_fg

    def MF_model_in_bin(self,lMlow,lMhigh,params):
        return integrate.quad(self.dndM_at_M,lMlow,lMhigh,args=(params))[0]*self.volume

    """
    The veriable 'variances' contains the
    variance in the paramters, namely f and g.

    The covariance between f and g is contained in the cov_fg term,
    which may or may not be passed in and if not is set to 0.
    """
    def var_MF_model_all_bins(self,lM_bins,params,variances,cov_fg = 0):
        lM_bins_natural = np.log(10**lM_bins)
        return np.array([self.var_MF_model_in_bin(lMlow,lMhigh,params,variances,cov_fg) for lMlow,lMhigh in lM_bins_natural])

    def MF_model_all_bins(self,lM_bins,params,redshift):
        if not (redshift == self.redshift):
            self.redshift = redshift
            self.scale_factor = 1./(1.+redshift)
            self.build_splines()
        lM_bins_natural = np.log(10**lM_bins)
        return np.array([self.MF_model_in_bin(lMlow,lMhigh,params) for lMlow,lMhigh in lM_bins_natural])

if __name__ == "__main__":
    #An example cosmology
    cosmo_dict = {"om":0.3,"ob":0.05,"ol":1.-0.3,\
                  "ok":0.0,"h":0.7,"s8":0.77,\
                  "ns":3.0,"w0":.96,"wa":0.0}
    bounds = np.log10([1e12,1e16]) #Mass bounds in Msun/h
    volume = 1e9 #(Mpc/h)^3
    redshift = 0.0

    MF = MF_model(cosmo_dict,bounds,volume,redshift)

    params = np.array([1.97,1.0,0.51,1.228]) #d,e,f,g

    bins = MF.make_bins(20,12,15)
    output = MF.MF_model_all_bins(bins,params,redshift)

    Masses = np.mean(10**bins,1)
    import matplotlib.pyplot as plt
    plt.loglog(Masses,output)
    plt.show()
