from __future__ import print_function
import argparse
from math import log10
#import skimage.io as skio

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Net
import Models
from data import get_training_set, get_test_set
from math import log10
from tensorboardX import SummaryWriter
import numpy as np
import os


parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--model', type=str, default='SRResNet', help='neural network model to use. Default=SRResNet')
parser.add_argument('--in_size', type=tuple, default=[14,14], help='dimensions of input image')
parser.add_argument('--res_channels', type=int, default=8, help='number of channels for residual layers. Default=64')
parser.add_argument('--num_dilations', type=int, default=1, help='number of dilations for residual layers. Default=1')
parser.add_argument('--num_shears', type=int, default=1, help='number of shears for residual layers. Default=1')
parser.add_argument('--model_save', type=str, default='', help='directory in which to save the model')
parser.add_argument('--load_path', type=str, default='', help='pth to load previously trained model ')
parser.add_argument('--np_load', type=str, default='', help='npy array of previous model to load ')

args = parser.parse_args()

torch.manual_seed(123)
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

paths = ["./ShearAndDilate_ch=2", "./ResNet_ch=2", "./ShearAndDilate_ch=2_NoShears", "./ShearAndDilate_ch=2_NoDilations", "./ShearAndDilate_ch=4", "./ResNet_ch=4", "./ShearAndDilate_ch=4_NoShears", "./ShearAndDilate_ch=4_NoDilations", "./ShearAndDilate_ch=8", "./ResNet_ch=8", "./ShearAndDilate_ch=8_NoShears", "./ShearAndDilate_ch=8_NoDilations", "./ShearAndDilate_ch=16", "./ShearAndDilate_ch=16_NoShears", "./ShearAndDilate_ch=16_NoDilations", "./ResNet_ch=16"] 

print('===> Loading datasets')
train_set = get_training_set(2)
test_set = get_test_set(2)
training_data_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=args.threads, batch_size=args.testBatchSize, shuffle=False)
model_list = ["SRResNet_shearAndDilate", "SRResNet"]

total_imgs= 0


for batch in training_data_loader:
			input, target = batch[0].to(device), batch[1].to(device)
			total_imgs += input.size()[0]


modelFns = {'SRResNet':Models.SRResNet.SRResNet, 'SRResNet_shear':Models.SRResNet_shear.SRResNet_shear, 'SRResNet_dilate':Models.SRResNet_dilate.SRResNet_dilate, 'SRResNet_shearAndDilate':Models.SRResNet_shearAndDilate.SRResNet_shearAndDilate}
modelFN = modelFns[ "SRResNet_shearAndDilate" ]

criterion = nn.MSELoss(reduction = 'mean')


def train_stats(epoch):
	epoch_loss = 0
	total = 0
	with torch.no_grad():
		for batch in training_data_loader:
			input, target = batch[0].to(device), batch[1].to(device)
			total += input.size()[0]
			loss = criterion(model(input), target)
			epoch_loss += loss.item()
	print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / (4*total)))
	return epoch_loss / float(4*total)


def test_stats():
	total = 0
	total_psnr = 0
	with torch.no_grad():
		for batch in testing_data_loader:
			input, target = batch[0].to(device), batch[1].to(device)
			total += input.size()[0]
			prediction = model(input)
			mse = (prediction - target)
			mse = mse*mse
			mse = torch.mean(torch.mean(mse, -1,), -1)
			mse = mse.clone().cpu().numpy()
			mse = 10*np.log10(1/mse)
			total_psnr += np.sum(mse)
	return total_psnr/float(4*total)


for i in range(len(paths) - 4):
	num_epochs = 200
	epoch_stats = np.zeros((2, num_epochs)) 	
	for epoch in range(num_epochs):
		model = torch.load(os.path.join(paths[i + 4], "checkpoints", "epoch_" + str(epoch + 1) + "=.pth") , map_location=device)
		model.eval()
		epoch_stats[0, epoch] = train_stats(epoch)
		epoch_stats[1, epoch] = test_stats()

	np.save(os.path.join(paths[i + 4], 'stats.npy'), epoch_stats)


