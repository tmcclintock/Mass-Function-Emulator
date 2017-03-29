"""
Instead of rotating the chains in the entire parameter space,
just rotate all the intercepts together and then all the slopes together.
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

outbase = "./mixed_chains/Mixed_Box%03d_chain.txt"

make_Rs = False
rotate = True

#GOT UP TO HERE AND STOPPED
if make_Rs:
    #First find all the rotation matrices
    for i in range(0,N_boxes):
        data = np.loadtxt(inbase%i)
        labs = ["int","slope"]
        for g in range(0,2): #First slopes, then intercepts
            D = data[:,g::2]
            C = np.cov(D,rowvar=False)
            w,R = np.linalg.eig(C)
            np.savetxt("./mixed_chains/R%s%d_matrix.txt"%(labs[g],i),R)
            #As it turns out, cosmo 34 is the middle-most box,
            #so use it for the rotation matrix.
            if i == 34: np.savetxt("./mixed_chains/R%s_matrix.txt"%labs[g],R)
            if i == 34: np.savetxt("./R%s_matrix.txt"%labs[g],R)
            if i == 34: np.savetxt("../R%s_matrix.txt"%labs[g],R)
            print "Created R%s%d"%(labs[g],i)

if rotate:
    #First get the Rotation matrix
    R = []
    R.append(np.loadtxt("./mixed_chains/Rint_matrix.txt"))
    R.append(np.loadtxt("./mixed_chains/Rslope_matrix.txt"))
    #Now rotate some chains
    for i in range(0,N_boxes):
        data = np.loadtxt(inbase%i)
        rD = np.zeros_like(data)
        for g in range(0,2):
            rD[:,g::2] = np.dot(data[:,g::2],R[g])
        np.savetxt(outbase%i,rD)
        mean_models[i] = np.mean(rD,0)
        var_models[i] = np.var(rD,0)
        print "Saved box%03d"%i
        #fig = corner.corner(data,labels=old_labels)
        #fig = corner.corner(rD)
        #plt.show()
        #sys.exit()
    np.savetxt("./mixed_dfg_means.txt",mean_models)
    np.savetxt("./mixed_dfg_vars.txt",var_models)
