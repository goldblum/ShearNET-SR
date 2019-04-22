import numpy as np
import matplotlib.pyplot as plt

channels = 4
train = 1

if train:
	stats = np.load("trained_models/ShearAndDilate_ch=" + str(channels) + "_1x/stats.npy")
	stats2 = np.load("trained_models/ResNet_ch=" + str(channels) + "/stats.npy")
	stats3 = np.load("trained_models/ShearAndDilate_ch=" + str(channels) + "_3d/stats.npy")

else:
	stats = np.load("trained_models/ShearAndDilate_ch=" + str(channels) + "_1x/stats2.npy")
	stats2 = np.load("trained_models/ResNet_ch=" + str(channels) + "/stats2.npy")
	stats3 = np.load("trained_models/ShearAndDilate_ch=" + str(channels) + "_3d/stats2.npy")


plt.plot(stats[0,:], label = "ShearNet kernels = " + str(channels))
plt.plot(stats2[0,:], label = "SRResNet kernels " + str(channels))
plt.plot(stats3[0,:], label = "ShearNet3d kernels = " + str(channels))

#plt.plot(0.034198185106620725*np.ones(len(stats[1,:])), label = "Average Bicubic")
plt.ylabel('mse')
plt.xlabel('epoch')
plt.xlim((25,200))
plt.ylim((.00025,.00125))
plt.legend(fontsize = 'x-large')
plt.tight_layout()

plt.show()

'''
314 - test

1001, 63 - train

0.034198185106620725
'''
