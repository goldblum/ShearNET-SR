import os 
import numpy as np
import skimage.io as skio

test_path = "./data/Multispectral/train"
test_label_path = "./data/Multispectral/train_labels"

def block_average(img, down_factor):
	m, n = np.shape(img)
	new_m = int(m/down_factor)
	new_n = int(n/down_factor)
	new_img = np.zeros((new_m,new_n))
	for i in range(new_m):
		for j in range(new_n):
			for k in range(down_factor):
				for l in range(down_factor):
					new_img[i,j] = new_img[i,j] + img[i*down_factor + k, j*down_factor + l]
			new_img[i,j] = new_img[i,j] / (down_factor ** 2)
	return np.float32(new_img)

for filename in os.listdir(test_label_path):
	img = skio.imread(os.path.join(test_label_path,filename))
	img0 = block_average(img, 2)
	img1 = block_average(img0, 2)
	skio.imsave(os.path.join(test_path,filename), img1)
	skio.imsave(os.path.join(test_label_path, filename), img0)
