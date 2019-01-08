from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
import skimage.io as skio
import numpy as np
from PIL import Image

from dataset import DatasetFromFolder
import torchvision.datasets as datasets


def download_mnist(dest="data"):
    output_image_dir = join(dest, "MNIST/")

    if not exists(output_image_dir):
        makedirs(dest)
        
        train_data = datasets.MNIST(root='./data', train=True, download=True, transform=None)


        train_label_path = join(dest, 'MNIST/train_labels/')
	train_path = join(dest, 'MNIST/train/')
	if not (exists(train_label_path) and exists(train_path)):
		makedirs(train_label_path)
		makedirs(train_path)
        for i in range(len(train_data)):
		img, digit = train_data[i]
		img0 = downsize(img, 2)
		img0.save(train_path + str(i) + '.png')
		img.save(train_label_path + str(i) + '.png')
		
	'''
        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)
	'''
        

	test_data = datasets.MNIST(root='./data', train=False, download=True, transform=None)


	test_label_path = join(dest, 'MNIST/test_labels/')
	test_path = join(dest, 'MNIST/test/')
	if not (exists(test_label_path) and exists(test_path)):
		makedirs(test_label_path)
		makedirs(test_path)
        for i in range(len(test_data)):
		img, digit = test_data[i]
		img0 = downsize(img, 2)
		img0.save(test_path + str(i) + '.png')
		img.save(test_label_path + str(i) + '.png')



        

    return output_image_dir

def downsize(img, down_factor):
	img = np.asarray(img)
	m, n = np.shape(img)
	new_m = int(np.floor(m/down_factor))
	new_n = int(np.floor(n/down_factor))
	new_img = np.zeros((new_m,new_n))
	for i in range(new_m):
		for j in range(new_n):
			new_img[i,j] = img[i*down_factor, j*down_factor]
	return (Image.fromarray(new_img)).convert("L")
def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor),
        ToTensor(),
    ])


def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])


def get_training_set(upscale_factor):
    root_dir = download_mnist()
    train_dir = join(root_dir, "train")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(train_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))


def get_test_set(upscale_factor):
    root_dir = download_mnist()
    test_dir = join(root_dir, "test")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(test_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))

