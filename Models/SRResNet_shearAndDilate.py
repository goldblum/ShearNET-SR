import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
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
		#print("dilated weights")
		#print(torch.matmul(torch.t(A), torch.matmul(weights, A)))
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
		new_weights = torch.transpose(new_weights, 2,3).contiguous()
	
	if pad:	
		new_weights = F.pad(new_weights, (shift,shift, shift, shift), "constant", 0)
		kernel_size = kernel_size + 2*shift
	while len(direction) < kernel_size: 
		direction.append("none")
		direction.insert(0, "none")
	Shear = np.zeros((kernel_size*kernel_size,kernel_size*kernel_size))
	for i in range(kernel_size):
		Shear[i*kernel_size:(i+1)*kernel_size, i*kernel_size:(i+1)*kernel_size] = shear_helper(kernel_size, direction[i], shift)
	Shear = torch.from_numpy(Shear).float().to(device)
	new_weights = new_weights.view(new_weights.shape[0], new_weights.shape[1], new_weights.shape[2] ** 2)
	if transpose:
		return torch.transpose(torch.matmul(new_weights, Shear).view(new_weights.shape[0], new_weights.shape[1], kernel_size, kernel_size), 2,3)
	else:
		return torch.matmul(new_weights, Shear).view(new_weights.shape[0], new_weights.shape[1], kernel_size, kernel_size)

def shear_manager(shear_num, dilation):
	direction = [["none"], ["up", "none", "down"],["down", "none", "up"],["up", "none", "down"],["down", "none", "up"]]
	dilated_direction = [["none"], ["up","none", "none","none", "down"],["down" , "none", "none", "none", "up"],["up","none", "none","none", "down"],["down" , "none", "none", "none", "up"]]
	shift = [0,1,1,1,1]
	transpose = [False, False, False, True, True]
	if dilation:
		return dilated_direction[shear_num], shift[shear_num], transpose[shear_num]
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
					
					direction, shift, transpose = shear_manager(j, i > 0)
					pad = calculate_padding(self.in_size, self.in_size, 1, dilated_kernel_size + 2*shift , 1)
					out = torch.cat((out, F.relu(F.conv2d(x, shear_weights(dilate_weights(self.conv_weight, i + 1, 3, self.device), dilated_kernel_size , direction, shift, transpose, self.device), bias=None, stride=1, padding=pad, dilation=1, groups=1))),1)		
		return out


class _Residual_Block(nn.Module):
    def __init__(self, in_size, device, res_trainable_channels):
        super(_Residual_Block, self).__init__()
        self.conv1 = shear_layer( res_trainable_channels * 10, in_size, res_trainable_channels, 2, 5, 3, device)
			#nn.Conv2d(res_trainable_channels, out_channels=res_trainable_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(res_trainable_channels*10, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = shear_layer( res_trainable_channels * 10, in_size, res_trainable_channels,  2, 5, 3, device)
			#nn.Conv2d(in_channels=res_trainable_channels, out_channels=res_trainable_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(res_trainable_channels*10, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output,identity_data)
        return output 

class Net(nn.Module):
    def __init__(self, device, res_trainable_channels = 8, im_size = 14):
        super(Net, self).__init__()
	self.res_trainable_channels = res_trainable_channels
	self.device = device
	self.im_size = im_size
        self.conv_input = nn.Conv2d(in_channels=4, out_channels = self.res_trainable_channels * 10, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.residual = self.make_layer(_Residual_Block, 16)
        self.conv_mid = nn.Conv2d(in_channels=self.res_trainable_channels * 10, out_channels=self.res_trainable_channels * 10, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(self.res_trainable_channels * 10, affine=True)
        self.upscale2x = nn.Sequential(
            nn.Conv2d(in_channels=self.res_trainable_channels * 10, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_output = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=9, stride=1, padding=4, bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(self.im_size, self.device, self.res_trainable_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out,residual)
        out = self.upscale2x(out)
        out = self.conv_output(out)
        return out


def SRResNet_shearAndDilate(device, trainable_channels):
	model = Net(device, trainable_channels)
	return model
