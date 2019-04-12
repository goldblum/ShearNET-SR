import torch.utils.data as data
import numpy as np
import skimage.io as skio
import torch
from os import listdir
from os.path import join
from PIL import Image


def is_image_file(filename):
	return any(filename.endswith(extension) for extension in [".tif"])


def load_img(filepath):
	endings = ["_GRE.tif", "_NIR.tif", "_RED.tif", "_REG.tif"]
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
		row, col = np.shape(skio.imread(self.image_filenames[0] + "_REG.tif"))
		input = load_img(self.image_filenames[index])
		target = load_img(self.label_filenames[index]) 
		means = [30322.45315498, 33028.37855216, 27555.90896633, 30824.52523748]
		std = [7555.26842064,  7995.16945722, 13374.68697786,  5576.38636]

		start_pixel = [np.random.randint(0, row - self.crop_size), np.random.randint(0, col - self.crop_size)]
		if self.input_transform:
			for i in range(self.num_channels):
				input[i] = self.input_transform((input[i] - means[i])/std[i], self.crop_size, start_pixel)
		if self.target_transform:
			for i in range(self.num_channels):
				target[i] = self.target_transform((target[i] - means[i])/std[i], self.crop_size * 2, [start_pixel[0]*2, start_pixel[1]*2])
		return torch.stack(input), torch.stack(target)

	def __len__(self):
		return len(self.image_filenames)

