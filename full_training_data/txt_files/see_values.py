import numpy as np

#Read in the input cosmologies
all_cosmologies = np.genfromtxt("../../test_data/building_cosmos_all_params.txt")
#all_cosmologies = np.delete(all_cosmologies,5,1) #Delete ln10As
all_cosmologies = np.delete(all_cosmologies,0,1) #Delete boxnum
all_cosmologies = np.delete(all_cosmologies,-1,0)#39 is broken
N_cosmologies = len(all_cosmologies)

means = np.genfromtxt("full_mean_models.txt")[:,2]
var = np.genfromtxt("full_var_models.txt")[:,2]
err = np.sqrt(var)
print means.shape
data = np.ones((N_cosmologies,2)) #Last column is for mean/erros
data[:,0] = means
data[:,1] = np.sqrt(var)

#Pick out the training/testing data
box_index, z_index = 0, 9
test_cosmo = all_cosmologies[box_index]
test_data = data[box_index]
training_cosmologies = np.delete(all_cosmologies,box_index,0)
training_data = np.delete(means,box_index,0)
training_errs = np.delete(err,box_index,0)
print training_cosmologies.shape
print test_data.shape

import emulator
emu = emulator.Emulator(name='e0',xdata=training_cosmologies,ydata=training_data,yerr=training_errs)
emu.train()

pred = emu.predict_one_point(test_cosmo)
print "prediction:",pred
print "truth:",test_data

import matplotlib.pyplot as plt
plt.errorbar(np.arange(len(means)),means,err)
plt.show()
