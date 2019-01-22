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
		A = torch.from_numpy(A).float().to(device)
		return torch.matmul(torch.t(A), torch.matmul(weights, A))

def shear_helper(kernel_size, direction, shift):
	sub_shear = np.zeros((kernel_size,kernel_size))
	if direction == "none" or shift == 0:
		return np.identity(kernel_size)
	if direction == "up":
		for i in range(kernel_size - shift):
			sub_shear[i, i + shift] = 1
	else:
		for i in range(kernel_size - shift):
			sub_shear[i + shift, i] = 1
	return sub_shear


def shear_weights(weights, kernel_size, direction, shift, transpose, device, pad = True):
	if shift == 0:
		return weights
	new_weights = weights.clone()
	if transpose:
		for i in range(new_weights.shape[0]):
			for j in range(new_weights.shape[1]):
				new_weights[i,j,:,:] = torch.t(new_weights[i,j,:,:])
	if pad:	
		new_weights = F.pad(weights, (shift,shift, shift, shift), "constant", 0)
		kernel_size = kernel_size + 2*shift
	while len(direction) < kernel_size: 
		direction.append("none")
		direction.insert(0, "none")
	Shear = np.zeros((kernel_size*kernel_size,kernel_size*kernel_size))
	for i in range(kernel_size):
		Shear[i*kernel_size:(i+1)*kernel_size, i*kernel_size:(i+1)*kernel_size] = shear_helper(kernel_size, direction[i], shift)
	Shear = torch.from_numpy(Shear).float().to(device)
	new_weights = torch.reshape(new_weights, (weights.shape[0], weights.shape[1], kernel_size ** 2))
	return torch.reshape(torch.matmul(new_weights, Shear),(new_weights.shape[0], new_weights.shape[1],kernel_size, kernel_size))

def shear_manager(shear_num):
	direction = [["none"], ["up", "none", "down"],["down", "none", "up"],["up", "none", "down"],["down", "none", "up"]]
	shift = [0,1,1,1,1]
	transpose = [False, False, False, True, True]
	return direction[shear_num], shift[shear_num], transpose[shear_num]
	

class shear_layer(nn.Module):
	def __init__(self, in_channels, in_size, out_channels, num_dilations, num_shears, kernel_size, device):
		super(shear_layer, self).__init__()
		self.in_channels = in_channels
		self.in_size = in_size
		self.out_channels = out_channels
		self.num_dilations = num_dilations #including no dilation
		self.num_shears = num_shears #including no shear
		self.kernel_size = kernel_size
		self.device = device
		self.conv_weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size).cuda())
		nn.init.orthogonal_(self.conv_weight, init.calculate_gain('relu'))
		


	def forward(self, x):
		pad = calculate_padding(self.in_size, self.in_size, 1, 3, 1)
		out = F.relu(F.conv2d(x, self.conv_weight, bias=None, stride=1, padding = pad , dilation=1, groups=1))
		for i in range(self.num_dilations):
			dilated_kernel_size = (i+1)*(self.kernel_size - 1) + 1
			for j in range(self.num_shears):
				if not (i + j == 0):
					direction, shift, transpose = shear_manager(j)
					pad = calculate_padding(self.in_size, self.in_size, 1, dilated_kernel_size + 2*shift , 1)
					out = torch.cat((out, F.relu(F.conv2d(x, shear_weights(dilate_weights(self.conv_weight, i + 1, 3, self.device), dilated_kernel_size , direction, shift, transpose, self.device), bias=None, stride=1, padding=pad, dilation=1, groups=1))),1)		
		return out

class Net(nn.Module):
	def __init__(self, device):
		super(Net, self).__init__()
		self.lay1 = shear_layer( 1, 14, 10, 2, 5, 3, device)
		self.lay2 = shear_layer( 100, 14, 4, 1, 1, 3, device)
		self.pixel_shuffle = nn.PixelShuffle(2)
		#init.orthogonal_(self.lay2.weight, init.calculate_gain('relu'))

	def forward(self, x):
		x = F.relu(self.lay1(x))
		x = F.relu(self.lay2(x)) 
		x = self.pixel_shuffle(x)
		return x

def SRResNet_shearAndDilate(device):
	model = Net(device)
	return model
