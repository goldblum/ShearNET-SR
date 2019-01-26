import os
from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
import skimage.io as skio
import numpy as np
from PIL import Image


def main():
	download_ms()

def download_ms(dest="./data"):
	output_image_dir = join(dest, "Multispectral/")

	if not exists(output_image_dir):
		makedirs(dest)
		

	train_label_path = join(dest, 'Multispectral/train_labels/')
	train_path = join(dest, 'Multispectral/train/')
	if not (exists(train_label_path) and exists(train_path)):
		makedirs(train_label_path)
		makedirs(train_path)

	for filename in os.listdir(train_path):
		img = skio.imread(join(train_path, filename))
		img_label = block_average(img, 2)
		img_train = block_average(img_label, 2)
		skio.imsave(train_label_path + filename, img_label)
		skio.imsave(train_path + filename, img_train)
		
	'''
		print("Extracting data")
		with tarfile.open(file_path) as tar:
			for item in tar:
				tar.extract(item, dest)
	'''
		

	test_label_path = join(dest, 'Multispectral/test_labels/')
	test_path = join(dest, 'Multispectral/test/')
	if not (exists(test_label_path) and exists(test_path)):
		makedirs(test_label_path)
		makedirs(test_path)

	for filename in os.listdir(test_path):
		img = skio.imread(os.join(test_path,filename))
		img_label = block_average(img, 2)
		img_train = block_average(img_label, 2)
		skio.imsave(test_label_path + filename, img_label)
		skio.imsave(test_path + filename, img_train)
		


		

	return output_image_dir

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

def downsize(img, down_factor):
	m, n = np.shape(img)
	new_m = int(np.floor(m/down_factor))
	new_n = int(np.floor(n/down_factor))
	new_img = np.zeros((new_m,new_n))
	for i in range(new_m):
		for j in range(new_n):
			new_img[i,j] = img[i*down_factor, j*down_factor]
	return np.float32(new_img)


main()
