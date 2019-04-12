import skimage.io as skio
import os
import numpy as np
from scipy import interpolate

train_label_path = './data/Multispectral/train_labels/'
train_path = './data/Multispectral/train/'

total = 0
total_psnr = 0

means = [30322.45315498, 33028.37855216, 27555.90896633, 30824.52523748]
std = [7555.26842064,  7995.16945722, 13374.68697786,  5576.38636]

def superResolveBicubic(img, upFactorBicubic, hist_match=False, epsilon=0.0, lowRes=[]):
	xShape, yShape = np.shape(img)
	x = np.arange(0, upFactorBicubic*xShape, upFactorBicubic)
	y = np.arange(0, upFactorBicubic*yShape, upFactorBicubic)
	f = interpolate.interp2d(x, y, np.transpose(img), kind='cubic')
	newX = np.arange(upFactorBicubic*xShape)
	newY = np.arange(upFactorBicubic*yShape)
	outImg = f(newX, newY)
	outImg = np.transpose(outImg)
	return outImg

i=0
for filename in sorted(os.listdir(train_path)):
	img = skio.imread(os.path.join(train_path, filename))
	img = (img - means[i%4]) / std[i%4]
	label = skio.imread(os.path.join(train_label_path, filename))
	label = (label - means[i%4]) / std[i%4]
	sr_img = superResolveBicubic(img, 2)
	mse = (sr_img - label)
	mse = np.multiply(mse, mse)
	mse = np.mean(mse)
	#max_val = max(np.amax(sr_img), np.amax(label))
	#max_val = max_val * max_val
	#mse = 10*np.log10(1/mse)
	total_psnr += mse
	total += 1
	i += 1
print(total_psnr/float(total))

'''
0.034198185106620725
'''

