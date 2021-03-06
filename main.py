from __future__ import print_function
import argparse
from math import log10
import skimage.io as skio

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

# Training settings
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

if args.cuda and not torch.cuda.is_available():
	raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

if args.np_load != '':
	epoch_stats = np.load(args.np_load)
else:
	epoch_stats = np.zeros((2, args.nEpochs))

print('===> Loading datasets')
train_set = get_training_set(2)
test_set = get_test_set(2)
training_data_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=args.threads, batch_size=args.testBatchSize, shuffle=False)

print('===> Building model')
writer = SummaryWriter()

modelFns = {'SRResNet':Models.SRResNet.SRResNet, 'SRResNet_shear':Models.SRResNet_shear.SRResNet_shear, 'SRResNet_dilate':Models.SRResNet_dilate.SRResNet_dilate, 'SRResNet_shearAndDilate':Models.SRResNet_shearAndDilate.SRResNet_shearAndDilate, 'SRResNet_shearAndDilate_3d':Models.SRResNet_shearAndDilate_3d.SRResNet_shearAndDilate_3d}
modelFN = modelFns[ args.model ]

if args.load_path != '':
	checkpoint = torch.load(args.load_path, map_location=device)
	model = modelFN(device, args.res_channels).to(device)
	model.load_state_dict(checkpoint['model'])
else: 
	model = modelFN(device, args.res_channels).to(device)

criterion = nn.MSELoss()

#model = torch.load('./checkpoints/SRResNet_shearAndDilate_epoch_91.pth', map_location = torch.device('cuda'))
optimizer = optim.Adam(list(model.parameters()), lr=args.lr)

def train(epoch):
	epoch_loss = 0
	total = 0
	for iteration, batch in enumerate(training_data_loader, 1):
		input, target = batch[0].to(device), batch[1].to(device)
		total += input.size()[0]
		'''
		if iteration == 50:
			skio.imsave("ainput_test.tif", torch.Tensor.cpu(torch.detach(input[0,0,:,:])).numpy())
			skio.imsave("atarget_test.tif", torch.Tensor.cpu(torch.detach(target[0,0,:,:])).numpy())
		'''
		optimizer.zero_grad()
		loss = criterion(model(input), target)
		'''
		if iteration == 50:
			skio.imsave("aoutput_test.tif", torch.Tensor.cpu(torch.detach(model(input))[0,0,:,:]).numpy())
			skio.imsave("atargetoutput_test.tif", torch.Tensor.cpu(torch.detach(target[0,0,:,:])).numpy())
		'''
		epoch_loss += loss.item()
		loss.backward()
		optimizer.step()
	print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / (4*total)))
	return epoch_loss / float(4*total)


def test(temp_model):
	total = 0
	total_psnr = 0
	with torch.no_grad():
		for batch in testing_data_loader:
			input, target = batch[0].to(device), batch[1].to(device)
			total += input.size()[0]
			#print(input[0])
			prediction = temp_model(input)
			#print(prediction[0])
			mse = (prediction - target)
			mse = mse*mse
			mse = torch.mean(torch.mean(mse, -1,), -1)
			mse = mse.clone().cpu().numpy()
			#mse = 10*np.log10(1/mse)
			total_psnr += np.sum(mse)
	return total_psnr/float(4*total)

def checkpoint(epoch):
	if args.model_save != '':
		if not os.path.isdir(args.model_save):
			os.mkdir(args.model_save)
			os.mkdir(os.path.join(args.model_save,"checkpoints"))		
	model_out_path = ("checkpoints/epoch_{}=.pth").format(epoch)
	torch.save(model, os.path.join(args.model_save, model_out_path))
	print("Checkpoint saved to {}".format(os.path.join(args.model_save, model_out_path)))

model_paths = ['ResNet_ch=2','ResNet_ch=4','ResNet_ch=8','ResNet_ch=16', './ShearAndDilate_ch=2_1x','./ShearAndDilate_ch=4_1x', './ShearAndDilate_ch=8_1x', './ShearAndDilate_ch=16_1x', './ShearAndDilate_ch=2_3d', './ShearAndDilate_ch=4_3d', './ShearAndDilate_ch=8_3d', './ShearAndDilate_ch=16_3d']

for i in range(len(model_paths)):
	for epoch in range(1, args.nEpochs + 1):
		load_path = model_paths[i] + '/checkpoints/epoch_' + str(epoch) + '=.pth'
		temp_model = torch.load(load_path)
		temp_model.to('cuda')
		#model = torch.load('./checkpoints/SRResNet_shearAndDilate_epoch_' + str(epoch) + '.pth', map_location = torch.device('cuda'))
		#epoch_stats[0, epoch -1] = train(epoch)
		epoch_stats[1, epoch-1] = test(temp_model)
		#checkpoint(epoch)
		np.save(os.path.join(model_paths[i], 'stats2.npy'), epoch_stats)
