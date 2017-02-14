import numpy as np

means = np.genfromtxt("full_mean_models.txt")[:,2]
var = np.genfromtxt("full_var_models.txt")[:,2]
err = np.sqrt(var)
print means.shape

import emulator
#emu = emulator.Emulator(name='e0',xdata=means,ydata=means

import matplotlib.pyplot as plt
plt.errorbar(np.arange(len(means)),means,err)
plt.show()
