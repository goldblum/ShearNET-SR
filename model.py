import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

def calculate_padding(input_size, output_size, stride, kernel_size, dilation):
	return int(((output_size - 1)*stride - input_size + kernel_size + (kernel_size - 1)*(dilation - 1))/2)

def dilate_weights(weights, dilation_factor, kernel_size, device):
	if dilation_factor == 1:
		return weights
	else: 
		A = np.identity(kernel_size)
		for i in range(kernel_size - 1):
			for j in range(dilation_factor - 1):
				A = np.insert(A, i + 1 + i*(dilation_factor - 1) + j, np.zeros(kernel_size), axis = 1)
		A = torch.from_numpy(A, device = device)
		A = A.type(torch.FloatTensor)
		return torch.matmul(torch.t(A), torch.matmul(weights, A))

def shear_weights(weights, kernel_size, shear, dilation_factor, device):
	Shear = np.identity(kernel_size)
	Shear[int(shear[0]), int(shear[1])] = shear[2]
	Shear = torch.from_numpy(Shear, device = device)
	Shear = Shear.type(torch.FloatTensor)
	return torch.matmul(weights, Shear)

class shear_layer(nn.Module):
	def __init__(self, in_channels, in_size, out_channels, num_dilations, shears, kernel_size, upscale_factor, device):
		super(shear_layer, self).__init__()
		self.in_channels = in_channels
		self.in_size = in_size
		self.out_channels = out_channels
		self.num_dilations = num_dilations
		self.shears = shears
		self.kernel_size = kernel_size
		self.upscale_factor = upscale_factor
		self.device = device
		

		self.conv_weight = nn.Parameter(torch.randn(out_channels/((num_dilations)*((np.shape(shears))[0])),in_channels, kernel_size, kernel_size)).cuda()
		nn.init.orthogonal_(self.conv_weight, init.calculate_gain('relu'))
		


	def forward(self, x):
		pad = calculate_padding(self.in_size,self.in_size, 1, 3, 1)
		out = F.relu(F.conv2d(x, self.conv_weight, bias=None, stride=1, padding = pad , dilation=1, groups=1))
		for i in range(self.num_dilations):
			dilated_kernel_size = (i+1)*(self.kernel_size - 1) + 1
			for j in range((np.shape(self.shears))[0]):
				if not (i + j == 0):
					pad = calculate_padding(self.in_size, self.in_size, 1, self.kernel_size, i+1) 
					out = torch.cat((out, F.relu(F.conv2d(x, shear_weights(dilate_weights(self.conv_weight, i + 1, 3, self.device), dilated_kernel_size , self.shears[j], i + 1, self.device), bias=None, stride=1, padding=pad, dilation=1, groups=1))),1)
		#x = self.pixel_shuffle(out)
		return out

class Net(nn.Module):
	def __init__(self, upscale_factor, device):
		super(Net, self).__init__()
		self.lay1 = shear_layer( 1, 14, 64, 2, [[0,0,1], [0,1,1]], 3, upscale_factor, device)
		self.lay2 = shear_layer( 64, 14, 4, 2, [[0,0,1], [0,1,1]], 3, upscale_factor, device)
		self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
		#init.orthogonal_(self.lay2.weight, init.calculate_gain('relu'))

	def forward(self, x):
		x = F.relu(self.lay1(x))
		x = F.relu(self.lay2(x)) 
		x = self.pixel_shuffle(x)
		return x
