import torch.utils.data as data
import numpy as np
import skimage.io as skio
import torch
import os
from os.path import join
from PIL import Image

path = "./data/Multispectral/"
means = np.zeros(4)
devs = np.zeros(4)
i=0
for filename in sorted(os.listdir(path)):
	x = skio.imread(os.path.join(path,filename))
	means[i % 4] += np.mean(x)
	devs[i % 4] += np.std(x)
	i += 1
means = means/float(len(os.listdir(path))/4)
devs = devs/float(len(os.listdir(path))/4)
print(means)
print(devs)
'''
for filename in os.listdir(path):
	os.rename(os. path.join(path, filename), os.path.join(path, filename.split(".")[0] + "." + filename.split(".")[2]))
'''
