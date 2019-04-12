import numpy as np
import matplotlib.pyplot as plt

channels = 8
train = 0

if train:
	stats = np.load("ShearAndDilate_ch=" + str(channels) + "_1x/stats.npy")
	stats2 = np.load("ResNet_ch=" + str(channels) + "/stats.npy")
	stats3 = np.load("ShearAndDilate_ch=" + str(channels) + "_3d/stats.npy")

else:
	stats = np.load("ShearAndDilate_ch=" + str(channels) + "_1x/stats2.npy")
	stats2 = np.load("ResNet_ch=" + str(channels) + "/stats2.npy")
	stats3 = np.load("ShearAndDilate_ch=" + str(channels) + "_3d/stats2.npy")


plt.plot(stats[1,:], label = "ShearNet ch = " + str(channels))
plt.plot(stats2[1,:], label = "ResNet ch = " + str(channels))
plt.plot(stats3[1,:], label = "ShearNet3d ch = " + str(channels))

plt.plot(0.034198185106620725*np.ones(len(stats[1,:])), label = "Average Bicubic")
plt.ylabel('mse')
plt.xlabel('epoch')
#plt.xlim((25,200))
#plt.ylim((.00025,.00125))
plt.legend(fontsize = 'x-large')
plt.tight_layout()

plt.show()

'''
314 - test

1001, 63 - train

0.034198185106620725
'''
