import numpy as np
import scipy.optimize as op
import emcee, sys, corner
import tinker_mass_function as TMF
sys.path.insert(0,'../visualization/')
import visualize
import training_likelihoods as TL

#Choose which modes to run
run_test = True
run_best_fit = True
run_mcmc = True
run_comparisons = True
see_corner = True

#MCMC configuration
N_parameters = 2
nwalkers, nsteps = 16, 500
nburn = 0 #500
#corner_labels = [r"$f0$",r"$f1$",r"$g0$",r"$g1$"]
corner_labels = [r"$f$",r"$g$"]

#These are the scale factors of the snapshots
scale_factors = np.array([0.25,0.333333,0.5,0.540541,0.588235,0.645161,0.714286,0.8,0.909091,1.0])
redshifts = 1./scale_factors - 1.0

#The volume of the building simulations
volume = (1050.)**3 #[Mpc/h]^3

#Gather the cosmological parameters
N_boxes, N_z = 39, 10 #39th is broken
all_cosmologies = np.genfromtxt("../cosmology_files/building_cosmos_all_params.txt")

#The paths to the data and covariances. This is hard coded in.
data_path = "/home/tmcclintock/Desktop/all_MF_data/building_MF_data/full_mf_data/Box%03d_full/Box%03d_full_Z%d.txt"
cov_path = "/home/tmcclintock/Desktop/all_MF_data/building_MF_data/covariances/Box%03d_cov/Box%03d_cov_Z%d.txt"

#Create the output files
from_scratch = True
if from_scratch:
    best_fit_models = np.zeros((N_boxes*N_z,N_parameters))
    np.savetxt("txt_files/best_fit_models.txt",best_fit_models)
    mean_models = np.zeros((N_boxes*N_z,N_parameters))
    np.savetxt("txt_files/mean_models.txt",mean_models)
    var_models = np.zeros((N_boxes*N_z,N_parameters))
    np.savetxt("txt_files/var_models.txt",var_models)
else: 
    best_fit_models = np.loadtxt("txt_files/best_fit_models.txt")
    mean_models = np.loadtxt("txt_files/mean_models.txt")
    var_models = np.loadtxt("txt_files/var_models.txt")

#Loop over cosmologies and redshifts
box_lo,box_hi = 0,1#0,N_boxes
z_lo,z_hi = 9,10#0,N_z #0,N_z
for i in xrange(box_lo,box_hi):
    #Get in the cosmology and create a cosmo_dict
    num,ombh2,omch2,w0,ns,ln10As,H0,Neff,sigma8 = all_cosmologies[i]
    h = H0/100.
    Ob,Om = ombh2/(h**2), ombh2/(h**2)+omch2/(h**2)
    cosmo_dict = {"om":Om,"ob":Ob,"ol":1-Om,"ok":0.0,"h":h,"s8":sigma8,"ns":ns,"w0":w0,"wa":0.0}
    
    for j in xrange(z_lo,z_hi):
        #Create a model object 
        z = redshifts[j] #get the redshift
        sf = scale_factors[j]
        MF_model = TMF.MF_model(cosmo_dict,z)

        #Get in the data
        lM_low,lM_high,N_data,NP = np.loadtxt(data_path%(i,i,j)).T
        cov = np.loadtxt(cov_path%(i,i,j))
        lM_bins = np.array([lM_low,lM_high]).T
        lM = np.log10(np.mean(10**lM_bins,1))
        #Remove bad data
        used = np.where(N_data > 0)[0]
        lM_bins = lM_bins[used]
        N_data = N_data[used]
        cov = cov[:,used]
        cov = cov[used,:]
        icov = np.linalg.inv(cov)

        #Guess the parameters
        guesses = np.array([1.97,1.0,0.51,1.228,1.0]) #d,e,f,g, ln_scatter
        guesses = np.array([0.51,1.228]) #f,g

        if run_test:
            test = TL.lnprob(guesses,sf,lM_bins,N_data,cov,icov,volume,MF_model)
            print "Test result = %f\n"%test

        if run_best_fit:
            nll = lambda *args:-TL.lnprob(*args)
            result = op.minimize(nll,guesses,args=(sf,lM_bins,N_data,cov,icov,volume,MF_model),method="Powell")
            best_fit_models[i*N_z+j] = result['x']
            print "Best fit for Box%03d Z%d:\n%s\n"%(i,j,result)
            if run_comparisons:
                f,g = result['x']
                d,e,f,g = 1.97,1.0,f,g
                MF_model.set_parameters(d,e,f,g)
                N = MF_model.n_in_bins(lM_bins)*volume
                N_err = np.sqrt(np.diagonal(cov))
                visualize.NM_plot(lM,N_data,N_err,lM,N)

        if run_mcmc:
            ndim = N_parameters
            start = best_fit_models[i*N_z+j]
            pos = [start + 1e-4*np.random.randn(ndim) for k in range(nwalkers)]
            sampler = emcee.EnsembleSampler(nwalkers,ndim,TL.lnprob,args=(sf,lM_bins,N_data,cov,icov,volume,MF_model))
            print "Performing MCMC on Box%03d Z%d"%(i,j)
            sampler.run_mcmc(pos,nsteps)
            print "MCMC complete for Box%03d Z%d\n"%(i,j)
            fullchain = sampler.flatchain
            likes = sampler.flatlnprobability
            burn = fullchain[:nwalkers*nburn]
            chain = fullchain[nwalkers*nburn:]
            np.savetxt("chains/Box%03d_Z%d_chain.txt"%(i,j),fullchain)
            np.savetxt("chains/Box%03d_Z%d_likes.txt"%(i,j),likes)
            mean_models[i*N_z+j] = np.mean(chain,0)
            var_models[i*N_z+j] = np.var(chain,0)
            if run_comparisons:
                f,g = np.mean(chain,0)
                d,e,f,g = 1.97,1.0,f,g
                MF_model.set_parameters(d,e,f,g)
                N = MF_model.n_in_bins(lM_bins)*volume
                N_err = np.sqrt(np.diagonal(cov))
                visualize.NM_plot(lM,N_data,N_err,lM,N)
        if see_corner:
            fullchain = np.loadtxt("chains/Box%03d_Z%d_chain.txt"%(i,j))
            chain = fullchain[nwalkers*nburn:]
            import matplotlib.pyplot as plt
            fig = corner.corner(chain,labels=corner_labels,plot_datapoints=False)
            plt.gcf().savefig("figures/Box%03d_Z%d_corner.png"%(i,j))
            plt.show()
            plt.close()
        continue #end loop over redshifts
    continue #end loop over boxes/cosmologies

#Save the models
header = "f\tg"
#header = "f0\tf1\tg0\g1"
np.savetxt("txt_files/best_fit_models.txt",best_fit_models,header=header)
np.savetxt("txt_files/mean_models.txt",mean_models,header=header)
np.savetxt("txt_files/var_models.txt",var_models,header=header)
