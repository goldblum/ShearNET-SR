import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

def main():
	

def shear_helper(kernel_size, direction, shift):
	sub_shear = np.zeros(kernel_size)
	if shift == 0:
		return np.identity(kernel_size)
	if direction == "up":
		for i in range(kernel_size - shift):
			sub_shear[i, i + shift] = 1
	else:
		for i in range(kernel_size - shift):
			sub_shear[i + shift, i] = 1
	return sub_shear
		
		
def shear_weights(weights, kernel_size, device):
	Shear = np.zeros(kernel_size*kernel_size)
	direction = ["up", "up", "down"]
	shift = [1, 0, -1]
	for i in range(kernel_size):
		Shear[i*kernel_size:(i+1)*kernel_size, i*kernel_size:(i+1)*kernel_size] = shear_helper(kernel_size, direction[i], shift[i])

	print(Shear)
	weights = torch.reshape(weights, (-1,))
	return torch.reshape(torch.mm(weights, Shear),(kernel_size, kernel_size))
	
	

