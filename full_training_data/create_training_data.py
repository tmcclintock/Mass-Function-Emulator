import numpy as np
import scipy.optimize as op
import emcee, sys, corner
import tinker_mass_function as TMF
sys.path.insert(0,'../visualization/')
import visualize
import training_likelihoods as TL

#Choose which modes to run
run_test = False
run_best_fit = False
run_bf_comparisons = False
run_mcmc = False
run_mcmc_comparisons = False
calculate_chi2 = False
see_corner = True

#MCMC configuration
N_parameters = 8
nwalkers, nsteps = 32, 2500
nburn = 750
corner_labels = [r"$d0$",r"$d1$",r"$e0$",r"$e1$",
                 r"$f0$",r"$f1$",r"$g0$",r"$g1$"]

#Scale factors, redshifts, volume
scale_factors = np.array([0.25,0.333333,0.5,0.540541,0.588235,0.645161,0.714286,0.8,0.909091,1.0])
redshifts = 1./scale_factors - 1.0
volume = 1050.**3 #[Mpc/h]^3

#Gather the cosmological parameters
N_boxes, N_z = 39, 10 #39th is broken
all_cosmologies = np.genfromtxt("../cosmology_files/building_cosmos_all_params.txt")

#The paths to the data and covariances. This is hard coded in.
data_path = "/home/tmcclintock/Desktop/all_MF_data/building_MF_data/full_mf_data/Box%03d_full/Box%03d_full_Z%d.txt"
cov_path = "/home/tmcclintock/Desktop/all_MF_data/building_MF_data/covariances/Box%03d_cov/Box%03d_cov_Z%d.txt"

#Create the output files
from_scratch = False
if from_scratch:
    best_fit_models = np.zeros((N_boxes,N_parameters))
    np.savetxt("txt_files/full_best_fit_models.txt",best_fit_models)
    mean_models = np.zeros((N_boxes,N_parameters))
    np.savetxt("txt_files/full_mean_models.txt",mean_models)
    var_models = np.zeros((N_boxes,N_parameters))
    np.savetxt("txt_files/full_var_models.txt",var_models)
    chi2s = np.zeros((N_boxes,N_z))
    np.savetxt("txt_files/full_chi2_models.txt",chi2s)
else: 
    best_fit_models = np.loadtxt("txt_files/full_best_fit_models.txt")
    mean_models = np.loadtxt("txt_files/full_mean_models.txt")
    var_models = np.loadtxt("txt_files/full_var_models.txt")
    chi2s = np.loadtxt("txt_files/full_chi2_models.txt")

#Loop over cosmologies and redshifts
box_lo,box_hi = 0,1#N_boxes
z_lo,z_hi = 0,N_z #Which redshifts to plot
for i in xrange(box_lo,box_hi):
    #Get in the cosmology and create a cosmo_dict
    num,ombh2,omch2,w0,ns,ln10As,H0,Neff,sigma8 = all_cosmologies[i]
    h = H0/100.
    Ob,Om = ombh2/(h**2), ombh2/(h**2)+omch2/(h**2)
    cosmo_dict = {"om":Om,"ob":Ob,"ol":1-Om,"ok":0.0,"h":h,"s8":sigma8,"ns":ns,"w0":w0,"wa":0.0}

    #Read in all of the mass functions
    lM_array = []
    lM_bin_array = []
    N_data_array = []
    cov_array = []
    icov_array = []
    MF_model_array = []
    for j in xrange(0,N_z):
        #Get in the data
        lM_low,lM_high,N_data,NP = np.loadtxt(data_path%(i,i,j)).T
        cov = np.loadtxt(cov_path%(i,i,j))
        lM_bins = np.array([lM_low,lM_high]).T
        #Remove bad data
        used = np.where(N_data > 0)[0]
        lM_bins = lM_bins[used]
        N_data = N_data[used]
        cov = cov[:,used]
        cov = cov[used,:]
        icov = np.linalg.inv(cov)
        #Add things to the arrays
        lM_array.append(np.log10(np.mean(10**lM_bins,1)))
        lM_bin_array.append(lM_bins)
        N_data_array.append(N_data)
        cov_array.append(cov)
        icov_array.append(icov)
        MF_model = TMF.MF_model(cosmo_dict,redshifts[j])
        MF_model_array.append(MF_model)
        continue
    
    #Guess the parameters
    guesses = np.array([1.97,1.0,0.51,1.228,-19.0]) #d,e,f,g, ln_scatter
    guesses = np.array([2.13,0.11,1.13,0.10, #d0,d1,e0,e1
                        0.41,0.15,1.25,0.11]) #f0,f1,g0,g1

    if run_test:
        test = TL.lnprob(guesses,scale_factors,redshifts,lM_bin_array,N_data_array,cov_array,icov_array,volume,MF_model_array)
        print "Test result = %f\n"%test

    if run_best_fit:
        nll = lambda *args:-TL.lnprob(*args)
        result = op.minimize(nll,guesses,args=(scale_factors,redshifts,lM_bin_array,N_data_array,cov_array,icov_array,volume,MF_model_array),method="Powell")
        best_fit_models[i] = result['x']
        print "Best fit for Box%03d:\n%s\n"%(i,result)

    if run_bf_comparisons:
        d0,d1,e0,e1,f0,f1,g0,g1 = best_fit_models[i]
        for j in range(z_lo,z_hi):
            d = d0+(scale_factors[j]-0.5)*d1
            e = e0+(scale_factors[j]-0.5)*e1
            f = f0+(scale_factors[j]-0.5)*f1
            g = g0+(scale_factors[j]-0.5)*g1
            MF_model_array[j].set_parameters(d,e,f,g)
            N = MF_model_array[j].n_in_bins(lM_bin_array[j])*volume
            N_err = np.sqrt(np.diagonal(cov_array[j]))
            visualize.NM_plot(lM_array[j],N_data_array[j],N_err,lM_array[j],N)
            
    if run_mcmc:
        ndim = N_parameters
        start = best_fit_models[i]
        pos = [start + 1e-2*np.random.randn(ndim) for k in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers,ndim,TL.lnprob,args=(scale_factors,redshifts,lM_bin_array,N_data_array,cov_array,icov_array,volume,MF_model_array))
        print "Performing MCMC on Box%03d"%(i)
        sampler.run_mcmc(pos,nsteps)
        print "MCMC complete for Box%03d\n"%(i)
        fullchain = sampler.flatchain
        likes = sampler.flatlnprobability
        burn = fullchain[:nwalkers*nburn]
        chain = fullchain[nwalkers*nburn:]
        np.savetxt("chains/Box%03d_chain.txt"%(i),fullchain)
        np.savetxt("chains/Box%03d_likes.txt"%(i),likes)
        mean_models[i] = np.mean(chain,0)
        var_models[i] = np.var(chain,0)

    if run_mcmc_comparisons:
        d0,d1,e0,e1,f0,f1,g0,g1 = mean_models[i]
        for j in range(z_lo,z_hi):
            d = d0+(scale_factors[j]-0.5)*d1
            e = e0+(scale_factors[j]-0.5)*e1
            f = f0+(scale_factors[j]-0.5)*f1
            g = g0+(scale_factors[j]-0.5)*g1
            MF_model_array[j].set_parameters(d,e,f,g)
            N = MF_model_array[j].n_in_bins(lM_bin_array[j])*volume
            N_err = np.sqrt(np.diagonal(cov_array[j]))
            sigdif = (N_data_array[j]-N)/N_err
            print "\nZ%d"%j
            for ind in range(len(N)):
                print "Bin %d: %.1f +- %.1f\tvs\t%.1f  at  %f"%(ind,N_data_array[j][ind],N_err[ind],N[ind],sigdif[ind])

            visualize.NM_plot(lM_array[j],N_data_array[j],N_err,lM_array[j],N,title="Box%03d at z=%.2f"%(i,redshifts[j]))

    if calculate_chi2:
        d0,d1,e0,e1,f0,f1,g0,g1 = mean_models[i]
        for j in range(z_lo,z_hi):
            d = d0+(scale_factors[j]-0.5)*d1
            e = e0+(scale_factors[j]-0.5)*e1
            f = f0+(scale_factors[j]-0.5)*f1
            g = g0+(scale_factors[j]-0.5)*g1
            MF_model_array[j].set_parameters(d,e,f,g)
            N_fit = MF_model_array[j].n_in_bins(lM_bin_array[j])*volume
            N_data = N_data_array[j]
            X = N_data-N_fit
            cov = cov_array[j]
            icov = np.linalg.inv(cov)
            chi2 = np.dot(X,np.dot(icov,X))
            chi2s[i,j] = chi2
        print "Chi2s for Box%03d are:"%i
        print chi2s[i]

    if see_corner:
        fullchain = np.loadtxt("chains/Box%03d_chain.txt"%(i))
        chain = fullchain[nwalkers*nburn:]
        import matplotlib.pyplot as plt
        fig = corner.corner(chain,labels=corner_labels,plot_datapoints=False)
        plt.gcf().savefig("figures/Box%03d_corner.png"%(i))
        plt.show()
        plt.close()

    #Save the models
    header = "d0\td1\te0\e1\tf0\tf1\tg0\tg1"
    np.savetxt("txt_files/full_best_fit_models.txt",best_fit_models,header=header)
    np.savetxt("txt_files/full_mean_models.txt",mean_models,header=header)
    np.savetxt("txt_files/full_var_models.txt",var_models,header=header)
    np.savetxt("txt_files/full_chi2_models.txt",chi2s)
    continue #end loop over boxes/cosmologies

