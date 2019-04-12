import numpy as np
from PIL import Image
import skimage.io as skio
import matplotlib.pyplot as plt 

def dilate_weights(weights, dilation_factor, kernel_size):
	if dilation_factor == 1:
		return weights
	else: 
		A = np.identity(kernel_size)
		for i in range(kernel_size - 1):
			for j in range(dilation_factor - 1):
				A = np.insert(A, i + 1 + i*(dilation_factor - 1) + j, np.zeros(kernel_size), axis = 1)
		return np.matmul(np.transpose(A), np.matmul(weights, A))

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


def shear_weights(weights, kernel_size, direction, shift, transpose, pad = True):
	if shift == 0:
		return weights
	new_weights = np.copy(weights)
	if transpose:	
		new_weights = np.transpose(new_weights)
	if pad:	
		new_weights = np.pad(new_weights, (shift, shift), "constant")
		kernel_size = kernel_size + 2*shift
	while len(direction) < kernel_size: 
		direction.append("none")
		direction.insert(0, "none")
	Shear = np.zeros((kernel_size*kernel_size,kernel_size*kernel_size))
	for i in range(kernel_size):
		Shear[i*kernel_size:(i+1)*kernel_size, i*kernel_size:(i+1)*kernel_size] = shear_helper(kernel_size, direction[i], shift)
	new_weights = np.reshape(new_weights, np.shape(new_weights)[0] ** 2)
	if transpose:
		return np.transpose(np.reshape(np.matmul(Shear, new_weights), (kernel_size, kernel_size)))
	else:
		return np.reshape(np.matmul(Shear, new_weights), (kernel_size, kernel_size))


def shear_manager(shear_num, dilation):
	direction = [["none"], ["up", "none", "down"],["down", "none", "up"],["up", "none", "down"],["down", "none", "up"]]
	dilated_direction = [["none"], ["up","none", "none","none", "down"],["down" , "none", "none", "none", "up"],["up","none", "none","none", "down"],["down" , "none", "none", "none", "up"]]
	shift = [0,1,1,1,1]
	transpose = [False, False, False, True, True]
	if dilation:
		return dilated_direction[shear_num], shift[shear_num], transpose[shear_num]
	return direction[shear_num], shift[shear_num], transpose[shear_num]
	

titles = ["base kernel", "x shear -1", "x shear +1", "y shear -1", "y shear +1", "dilation", "dilation \n x shear -1", "dilation \n x shear +1", "dilation \n y shear -1", "dilation \n y shear +1"]
kernel_size = 3
kernel = np.identity(3)
fig = plt.figure()
ax = fig.add_subplot(2,5,1)
ax.set_title(titles[0], fontsize = 8)
plt.imshow(kernel, cmap = plt.cm.gray)
for i in range(2):
		dilated_kernel_size = (i+1)*(3 - 1) + 1
		for j in range(5):
			if not (i + j == 0):
				direction, shift, transpose = shear_manager(j, i > 0)
				out = shear_weights(dilate_weights(kernel, i + 1, 3), dilated_kernel_size , direction, shift, transpose)
				ax = fig.add_subplot(2,5,j + i*5 + 1)
				ax.set_title(titles[j + i*5], fontsize = 8)
				plt.imshow(out, cmap = plt.cm.gray)
#plt.subplots_adjust(wspace = 0.75, hspace = 0.001)
plt.subplots_adjust(hspace = 0.000001)
plt.show()

