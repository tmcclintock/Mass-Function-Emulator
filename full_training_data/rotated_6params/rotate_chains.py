"""
Here, take the chains already computed and rotate them to break tight correlations.
"""
import numpy as np
import corner, sys
import matplotlib.pyplot as plt

old_labels = [r"$d0$",r"$d1$",r"$f0$",r"$f1$",r"$g0$",r"$g1$"]

N_z = 10
N_boxes = 39
N_p = 6
mean_models = np.zeros((N_boxes,N_p))
var_models = np.zeros((N_boxes,N_p))

#Just use Box000 to find the rotations
index = 0
inbase = "../6params/chains/Box%03d_chain.txt"

outbase = "./rotated_chains/Rotated_Box%03d_chain.txt"

make_Rs = True
rotate = True

if make_Rs:
    #First find all the rotation matrices
    for i in range(0,N_boxes):
        data = np.loadtxt(inbase%i)
        D = np.copy(data)
        C = np.cov(D,rowvar=False)
        w,R = np.linalg.eig(C)
        np.savetxt("./rotated_chains/R%d_matrix.txt"%i,R)
        #As it turns out, cosmo 34 is the middle-most box,
        #so use it for the rotation matrix.
        if i == 34: np.savetxt("./rotated_chains/R_matrix.txt",R)
        if i == 34: np.savetxt("./R_matrix.txt",R)
        if i == 34: np.savetxt("../R_matrix.txt",R)
        print "Created R%d"%i

if rotate:
    #First get the Rotation matrix
    R_matrices = []
    for i in range(0,N_boxes):
        R_matrices.append(np.loadtxt("./rotated_chains/R%d_matrix.txt"%i))
    R_matrices = np.array(R_matrices)
    #R = np.mean(R_matrices,0)
    R = np.loadtxt("./rotated_chains/R_matrix.txt")
    #Now rotate some chains
    for i in range(0,N_boxes):
        data = np.loadtxt(inbase%i)
        imeans = np.mean(data,0)
        rD = np.dot(data[:],R)
        np.savetxt(outbase%i,rD)
        mean_models[i] = np.mean(rD,0)
        var_models[i] = np.var(rD,0)
        print "Saved box%03d"%i
        #fig = corner.corner(data,labels=old_labels)
        #fig = corner.corner(rD)
        #plt.show()
    np.savetxt("./rotated_dfg_means.txt",mean_models)
    np.savetxt("./rotated_dfg_vars.txt",var_models)
    #np.savetxt("./R_matrix.txt",R)
