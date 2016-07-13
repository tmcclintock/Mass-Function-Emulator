"""
This file takes the full cosmology file
for the building boxes
 and creates smaller files with A_s
removed and the box numbers removed,
as well as scale factors added on.
"""

import numpy as np

"""
Input format:
boxnum ombh2 omch2 w0 ns ln10As H0 Neff sigma8
"""
full_cosmos = np.genfromtxt("building_cosmos_all_params.txt")
used_cosmos = full_cosmos[:,1:]

#used_cosmos = np.delete(used_cosmos,4,1) #Delete ln10As

np.savetxt("building_cosmos_no_z.txt",used_cosmos)

"""
Now make a 400 length array with scale factors added on to the end.
"""
scale_factors = np.array([0.25,0.333333,0.5,0.540541,0.588235,0.645161,0.714286,0.8,0.909091,1.0])

z_cosmos = []
for i in xrange(0,len(used_cosmos)):
    for j in xrange(0,len(scale_factors)):
        z_cosmos.append(np.append(used_cosmos[i],scale_factors[j]))

z_cosmos = np.array(z_cosmos)
print z_cosmos.shape

np.savetxt("building_cosmos_with_z.txt",z_cosmos)
