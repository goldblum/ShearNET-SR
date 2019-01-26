import torch.utils.data as data
import numpy as np
import skimage.io as skio
import torch
from os import listdir
from os.path import join
from PIL import Image


def is_image_file(filename):
	return any(filename.endswith(extension) for extension in [".TIF"])


def load_img(filepath):
	endings = ["_GRE.TIF", "_NIR.TIF", "_RED.TIF", "_REG.TIF"]
	return [skio.imread(filepath + ending) for ending in endings] 


class DatasetFromFolder(data.Dataset):
	def __init__(self, image_dir, label_dir, input_transform, target_transform, num_channels = 4, crop_size = 100):
		super(DatasetFromFolder, self).__init__()
		
		self.image_filenames = [join(image_dir, x.split("_")[0]) for x in listdir(image_dir) if is_image_file(x)]
		self.image_filenames = sorted(list(set(self.image_filenames)))
		self.label_filenames = [join(label_dir, x.split("_")[0]) for x in listdir(label_dir) if is_image_file(x)]
		self.label_filenames = sorted(list(set(self.label_filenames)))
		self.num_channels = num_channels
		self.crop_size = crop_size
		self.input_transform = input_transform
		self.target_transform = target_transform

	def __getitem__(self, index):
		row, col = np.shape(skio.imread(self.image_filenames[0] + "_REG.TIF"))
		input = load_img(self.image_filenames[index])
		target = load_img(self.label_filenames[index]) 
		start_pixel = [np.random.randint(0, row - self.crop_size), np.random.randint(0, col - self.crop_size)]
		#print("start pixel", start_pixel, [start_pixel[0]*2, start_pixel[1]*2], "crop size", self.crop_size, self.crop_size * 2)
		if self.input_transform:
			for i in range(self.num_channels):
				input[i] = self.input_transform(input[i], self.crop_size, start_pixel)
		if self.target_transform:
			for i in range(self.num_channels):
				target[i] = self.target_transform(target[i], self.crop_size * 2, [start_pixel[0]*2, start_pixel[1]*2])
		return torch.stack(input), torch.stack(target)

	def __len__(self):
		return len(self.image_filenames)

