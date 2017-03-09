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
inbase = "chains/Box%03d_chain.txt"
indata = np.loadtxt(inbase%index)
print indata.shape

outdata = np.copy(indata)
outbase = "rotated_chains/Rotated_Box%03d_chain.txt"

indices = [0,1,2,3,4,5] #Indices to rotate
D = np.copy(indata)
C = np.cov(D,rowvar=False)
w,R = np.linalg.eig(C)
np.savetxt("rotated_chains/R_matrix.txt",R)
for i in range(0,N_boxes):#3 boxes
    data = np.loadtxt(inbase%i)
    rD = np.dot(data[:],R)
    np.savetxt(outbase%i,rD)
    mean_models[i] = np.mean(rD,0)
    var_models[i] = np.var(rD,0)
    print "Saved box%03d"%i
np.savetxt("txt_files/rotated_mean_models.txt",mean_models)
np.savetxt("txt_files/rotated_var_models.txt",var_models)
np.savetxt("txt_files/R_matrix.txt",R)
