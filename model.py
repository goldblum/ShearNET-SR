import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

def calculate_padding(input_size, output_size, stride, kernel_size, dilation):
	return int(((output_size - 1)*stride - input_size + kernel_size + (kernel_size - 1)*(dilation - 1))/2)

def dilate_weights(weights, dilation_factor, kernel_size):
	if dilation_factor == 1:
		return weights
	else: 
		A = np.identity(kernel_size)
		for i in range(kernel_size - 1):
			for j in range(dilation_factor):
				A = np.insert(A, i + j, np.zeros(kernel_size), axis = 1)
			print(A)
		A = torch.from_numpy(A)
		A = A.type(torch.FloatTensor)
		blah = torch.matmul(torch.t(A), torch.matmul(weights, A))
		print(blah.shape)
		return torch.matmul(torch.t(A), torch.matmul(weights, A))

def shear_weights(weights, shear_parameter, kernel_size, shear_position):
	Shear = np.identity(kernel_size)
	Shear[int(np.floor(shear_position/kernel_size)), shear_position % kernel_size] = 1
	Shear = torch.from_numpy(Shear)
	Shear = Shear.type(torch.FloatTensor)
	return torch.matmul(Shear, weights)

class Net(nn.Module):
	def __init__(self, upscale_factor):
		super(Net, self).__init__()
		
		self.num_dilations = 2
		self.num_shears = 2
		self.kernel_size = 3
		self.relu = nn.ReLU()
		self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
		self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))

		self.conv_weight = nn.Parameter(torch.randn(64/((self.num_dilations)*(self.num_shears)),64, 3, 3))
		
		self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
		self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
		self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

		self._initialize_weights()

	def forward(self, x):
		x = self.relu(self.conv1(x))
		x = self.relu(self.conv2(x)) 
		pad = calculate_padding(128,128, 1, 3, 1)
		out = F.relu(F.conv2d(x, self.conv_weight, bias=None, stride=1, padding = pad , dilation=1, groups=1))
		for i in range(self.num_dilations):
			for j in range(self.num_shears):
				if not (i * j == 1):
					pad = calculate_padding(128, 128, 1, self.kernel_size, (i + 1)*self.kernel_size -1) 
					out = torch.cat((out, F.relu(F.conv2d(x, shear_weights(dilate_weights(self.conv_weight, i + 1, 3), 1, (i+1)*kernel_size, j + 1), bias=None, stride=1, padding=pad, dilation=1, groups=1))),1)
		x = self.relu(self.conv3(out))
		x = self.pixel_shuffle(self.conv4(x))
		return x

	def _initialize_weights(self):
		init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
		init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
		init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
		init.orthogonal_(self.conv4.weight)

    

	

		
