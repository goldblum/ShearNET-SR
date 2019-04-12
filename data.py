from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
import torch
import skimage.io as skio
import numpy as np
from PIL import Image

from dataset import DatasetFromFolder
import torchvision.datasets as datasets

def download_ms(dest="./data"):
	output_image_dir = join(dest, "Multispectral/")
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

def image_crop(img, crop_size, start_pixel):
	start_x = start_pixel[0]
	start_y = start_pixel[1]
	return img[start_x:start_x+crop_size, start_y:start_y+crop_size]


def input_transform(img, crop_size, start_pixel):
	return torch.from_numpy(img)
	#return torch.from_numpy(image_crop(img, crop_size, start_pixel))


def target_transform(img, crop_size, start_pixel):
	return torch.from_numpy(img)
	#return torch.from_numpy(image_crop(img, crop_size, start_pixel))

def get_training_set(upscale_factor):
	root_dir = download_ms()
	train_dir = join(root_dir, "train")
	train_label_dir = join(root_dir, "train_labels")

	return DatasetFromFolder(train_dir,train_label_dir, input_transform, target_transform)


def get_test_set(upscale_factor):
	root_dir = download_ms()
	test_dir = join(root_dir, "test")
	test_label_dir = join(root_dir, "test_labels")

	return DatasetFromFolder(test_dir, test_label_dir, input_transform, target_transform)


