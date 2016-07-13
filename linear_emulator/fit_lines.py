
import numpy as np
import matplotlib.pyplot as plt

scale_factors = np.array([0.25,0.333333,0.5,0.540541,0.588235,0.645161,0.714286,0.8,0.909091,1.0])
redshifts = 1./scale_factors - 1.0

data_path = "../building_data/building_means_all_z.txt"
vars_path = "../building_data/building_vars_all_z.txt"

data = np.genfromtxt(data_path)
variances = np.genfromtxt(vars_path)
f_all,g_all = data[:,0],data[:,1]
f_var_all,g_var_all = variances[:,0],variances[:,1]

make_f_plots = False
make_g_plots = False

#First fit f with a line for each cosmology
N_cosmos = 39 #40 #last one is broken
N_reds = 10
f0_path = "linear_fits/f0.txt"
f1_path = "linear_fits/f1.txt"
f0_var_path = "linear_fits/f0_var.txt"
f1_var_path = "linear_fits/f1_var.txt"
f0 = []
f0_var = []
f1 = []
f1_var = []
for i in xrange(0,N_cosmos):
    #Loop over redshifts to assemble the correct vector
    f_i = f_all[i*N_reds:i*N_reds+10]
    fv_i = f_var_all[i*N_reds:i*N_reds+10]

    ones = np.ones_like(scale_factors)
    A = np.vstack((ones,scale_factors))
    C = np.diag(fv_i)
    Cinv = np.linalg.inv(C)
    left = np.linalg.inv(np.dot(A,np.dot(Cinv,A.T)))
    right = np.dot(A,np.dot(Cinv,f_i))
    b,m = np.dot(left,right)
    cout = np.linalg.inv(np.dot(A,np.dot(Cinv,A.T)))
    #print "i = %d"%i
    #print "f0 = %f +- %f"%(b,np.sqrt(cout[0,0]))
    #print "f1 = %f +- %f"%(m,np.sqrt(cout[1,1]))
    #print ""
    f0.append(b)
    f0_var.append(cout[0,0])
    f1.append(m)
    f1_var.append(cout[1,1])
    if make_f_plots:
        domain = np.linspace(min(scale_factors),max(scale_factors),100)
        model = b+m*domain
        plt.errorbar(scale_factors,f_i,np.sqrt(fv_i))
        plt.plot(domain,model)
        plt.show()
f0 = np.array(f0)
f1 = np.array(f1)
f0_var = np.array(f0_var)
f1_var = np.array(f1_var)

np.savetxt(f0_path,f0)
np.savetxt(f1_path,f1)
np.savetxt(f0_var_path,f0_var)
np.savetxt(f1_var_path,f1_var)

g0_path = "linear_fits/g0.txt"
g1_path = "linear_fits/g1.txt"
g0_var_path = "linear_fits/g0_var.txt"
g1_var_path = "linear_fits/g1_var.txt"
g0 = []
g0_var = []
g1 = []
g1_var = []
for i in xrange(0,N_cosmos):
    #Loop over redshifts to assemble the correct vector
    g_i = g_all[i*N_reds:i*N_reds+10]
    gv_i = g_var_all[i*N_reds:i*N_reds+10]

    ones = np.ones_like(scale_factors)
    A = np.vstack((ones,scale_factors))
    C = np.diag(gv_i)
    Cinv = np.linalg.inv(C)
    left = np.linalg.inv(np.dot(A,np.dot(Cinv,A.T)))
    right = np.dot(A,np.dot(Cinv,g_i))
    b,m = np.dot(left,right)
    cout = np.linalg.inv(np.dot(A,np.dot(Cinv,A.T)))
    print "i = %d"%i
    print "g0 = %g +- %g"%(b,np.sqrt(cout[0,0]))
    print "g1 = %g +- %g"%(m,np.sqrt(cout[1,1]))
    print ""
    g0.append(b)
    g0_var.append(cout[0,0])
    g1.append(m)
    g1_var.append(cout[1,1])
    if make_g_plots:
        domain = np.linspace(min(scale_factors),max(scale_factors),100)
        model = b+m*domain
        plt.errorbar(scale_factors,g_i,np.sqrt(gv_i))
        plt.plot(domain,model)
        plt.show()
g0 = np.array(g0)
g1 = np.array(g1)
g0_var = np.array(g0_var)
g1_var = np.array(g1_var)

np.savetxt(g0_path,g0)
np.savetxt(g1_path,g1)
np.savetxt(g0_var_path,g0_var)
np.savetxt(g1_var_path,g1_var)


    
