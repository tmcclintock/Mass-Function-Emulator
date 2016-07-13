"""
This file reads in the chains and creates
the f,g, f_var, and g_var statistics
to be used by the emulator.
"""

import numpy as np

create_all = False
create_z_data = True

Nfiles = 39
Nreds = 10

path = "mcmc_chains/box%03d_Z%d_chain.txt"

"""
First create the full output data,
with 
"""
if create_all:
    means_all_z = []
    vars_all_z = []
    for i in xrange(0,Nfiles):
        for j in xrange(0,Nreds):
            chain = np.genfromtxt(path%(i,j))
            means_all_z.append(np.mean(chain,0))
            vars_all_z.append(np.var(chain,0))

    means_all_z = np.array(means_all_z)
    vars_all_z = np.array(vars_all_z)
    #print means_all_z
    print means_all_z.shape, vars_all_z.shape
    np.savetxt("building_data/building_means_all_z.txt",means_all_z)
    np.savetxt("building_data/building_vars_all_z.txt",vars_all_z)

if create_z_data:
    means_all_z = np.loadtxt("building_data/building_means_all_z.txt")
    vars_all_z = np.loadtxt("building_data/building_vars_all_z.txt")
    for j in xrange(0,Nreds):
        means_one_z = []
        vars_one_z = []
        for i in xrange(0,Nfiles):
            means_one_z.append(means_all_z[i*Nreds+j])
            vars_one_z.append(vars_all_z[i*Nreds+j])
        means_one_z = np.array(means_one_z)
        vars_one_z = np.array(vars_one_z)
        np.savetxt("building_data/building_means_Z%d.txt"%j,means_one_z)
        np.savetxt("building_data/building_vars_Z%d.txt"%j,vars_one_z)
