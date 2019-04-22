from __future__ import print_function
import argparse
from math import log10
import skimage.io as skio
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Net
import Models
from data import get_training_set, get_test_set
from math import log10
from tensorboardX import SummaryWriter
import numpy as np
from scipy import interpolate
import os
from skimage.measure import compare_ssim as ssim


model_paths = ['ResNet_ch=2','ResNet_ch=4','ResNet_ch=8','ResNet_ch=16', './ShearAndDilate_ch=2_1x','./ShearAndDilate_ch=4_1x', './ShearAndDilate_ch=8_1x', './ShearAndDilate_ch=16_1x', './ShearAndDilate_ch=2_3d', './ShearAndDilate_ch=4_3d', './ShearAndDilate_ch=8_3d', './ShearAndDilate_ch=16_3d']

print('===> Loading datasets')
test_set = get_test_set(2)
testing_data_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=10, shuffle=False)

device = 'cuda'
channels = 8
load_path1 = 'trained_models/ResNet_ch=' + str(channels) + '/checkpoints/epoch_200=.pth'
load_path2 = 'trained_models/ShearAndDilate_ch=' + str(channels) + '_1x/checkpoints/epoch_200=.pth'
load_path3 = 'trained_models/ShearAndDilate_ch=' + str(channels) + '_3d/checkpoints/epoch_200=.pth'

model1 = torch.load(load_path1)
model2 = torch.load(load_path2)
model3 = torch.load(load_path3)
model1.to(device)
model2.to(device)
model3.to(device)

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

def upsample(img, upFactor):
	xShape, yShape = np.shape(img)
	out_img = np.zeros((upFactor*xShape, upFactor*yShape))
	for i in range(xShape):
		for j in range(yShape):
			for k in range(upFactor):
				for l in range(upFactor):
					out_img[upFactor*i + k, upFactor*j + l] = img[i, j]
	return out_img

img_num = 2
img_channel = 0

i = 0
whole_image = 0
if whole_image:	
	row = [0, 480]
	col = [0, 640]
else:
	row = [30, 125]
	col = [35,140]
bound = 1
total_ssim1 = 0
total_ssim2 = 0
total_ssim3 = 0
total_ssim4 = 0
total = 0
ssim_ratio = 0
batch_point = 0
image_point = 0
channel_point = 0
batch_num = 28
with torch.no_grad():
	for batch in testing_data_loader:
		if i == batch_num:
			input, target = batch[0].to(device), batch[1].to(device)
			prediction1 = model1(input)
			prediction2 = model2(input)
			prediction3 = model3(input)			
			img = (target.clone().cpu().numpy())[img_num, img_channel,row[0]:row[1], col[0]:col[1]]
			img1 = (prediction1.clone().cpu().numpy())[img_num, img_channel,row[0]:row[1], col[0]:col[1]]
			img2 = (prediction2.clone().cpu().numpy())[img_num, img_channel,row[0]:row[1], col[0]:col[1]]
			img3 = (prediction3.clone().cpu().numpy())[img_num, img_channel,row[0]:row[1], col[0]:col[1]]
			img4 = superResolveBicubic((input.clone().cpu().numpy())[img_num, img_channel,:,:],2)[row[0]:row[1], col[0]:col[1]]
			img5 = upsample((input.clone().cpu().numpy())[img_num, img_channel,:,:],2)[row[0]:row[1], col[0]:col[1]]
			break

		'''
		for image in range(input.size()[1]):
			for chan_num in range(input.size()[1]):
					ssim1 = ssim(prediction1[image, chan_num, :, :],target[image, chan_num, :,:],win_size=None, data_range=1-0)
					ssim2 = ssim(prediction2[image, chan_num, :, :],target[image, chan_num, :,:],win_size=None, data_range=1-0)
					if ssim2/ssim1 > ssim_ratio:
						ssim_ratio = ssim2/ssim1
						batch_point = i
						image_point = image
						channel_point = chan_num
		
		'''
		'''
		for num_img in range(input.size()[0]):
			for chan_num in range(input.size()[1]):
				total_ssim1 += ssim(prediction1[num_img, chan_num, :,:], target[num_img, chan_num,:,:].clone().cpu().numpy())
				total_ssim2 += ssim(prediction2[num_img, chan_num, :,:], target[num_img, chan_num,:,:].clone().cpu().numpy())
				total_ssim3 += ssim(prediction3[num_img, chan_num, :,:], target[num_img, chan_num,:,:].clone().cpu().numpy())
		total += input.size()[0]*input.size()[1]
		'''
		i += 1
		print(i)

if bound:
	base = plt.imshow((target.clone().cpu().numpy())[img_num, img_channel,:,:])
	plt.plot(col[0]*np.ones(row[1]-row[0]),range(row[0], row[1]), '--', linewidth=1, color='firebrick')
	plt.plot(col[1]*np.ones(row[1]-row[0]),range(row[0], row[1]), '--', linewidth=1, color='firebrick')
	plt.plot(range(col[0], col[1]), row[0]*np.ones(col[1]-col[0]), '--', linewidth=1, color='firebrick')
	plt.plot(range(col[0], col[1]),row[1]*np.ones(col[1]-col[0]), '--', linewidth=1, color='firebrick')
	plt.show()
'''
print(total_ssim1/float(total), total_ssim2/float(total), total_ssim3/float(total))
'''

'''
	fig, ax = plt.subplots(nrows = 2, ncols = 2)
	ax[0,0].imshow(img)
	ax[0,0].set_title('HiRes', fontsize = 12)
	ax[0,1].imshow(img2)
	ax[0,1].set_title('ShearNet', fontsize = 12)
	ax[1,0].imshow(img4)
	ax[1,0].set_title('Bicubic', fontsize = 12)
	ax[1,1].imshow(img5)
	ax[1,1].set_title('LowRes', fontsize = 12)
	plt.tight_layout()
	plt.show()

'''

fig, ax = plt.subplots(nrows = 1, ncols = 3)
ax[0].imshow(img)
ax[0].set_title('HiRes')
ax[2].imshow(img1)
ax[2].set_title('SRResNet')
ax[1].imshow(img2)
ax[1].set_title('ShearNet')
plt.tight_layout()
plt.show()

	
'''
fig, ax = plt.subplots(nrows = 2, ncols = 3)
ax[0,0].imshow(img)
ax[0,0].set_title('HiRes')
ax[1,0].imshow(img1)
ax[1,0].set_title('SRResNet')
ax[0,1].imshow(img2)
ax[0,1].set_title('ShearNet')
ax[0,2].imshow(img3)
ax[0,2].set_title('ShearNet3d')
ax[1,1].imshow(img4)
ax[1,1].set_title('Bicubic')
ax[1,2].imshow(img5)
ax[1,2].set_title('LowRes')
plt.tight_layout()
plt.show()
'''
	

