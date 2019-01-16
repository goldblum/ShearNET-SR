from __future__ import print_function
import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Net
import Models
from data import get_training_set, get_test_set

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--model', type=str, default='SRResNet', help='neural network model to use. Default=SRResNet')
parser.add_argument('--in_size', type=tuple, default=[14,14], help='dimensions of input image')
parser.add_argument('--res_channels', type=int, default=64, help='number of channels for residual layers. Default=64')
parser.add_argument('--num_dilations', type=int, default=1, help='number of dilations for residual layers. Default=1')
parser.add_argument('--num_shears', type=int, default=1, help='number of shears for residual layers. Default=1')
args = parser.parse_args()

if args.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

print('===> Loading datasets')
train_set = get_training_set(2)
test_set = get_test_set(2)
training_data_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=args.threads, batch_size=args.testBatchSize, shuffle=False)

print('===> Building model')

modelFns = {'SRResNet':Models.SRResNet.SRResNet, 'SRResNet_shear':Models.SRResNet_shear.SRResNet_shear, 'SRResNet_dilate':Models.SRResNet_dilate.SRResNet_dilate, 'SRResNet_shearAndDilate':Models.SRResNet_shearAndDilate.SRResNet_shearAndDilate}
modelFN = modelFns[ args.model ]
model = modelFN(in_size, res_channels, num_dilations, num_shears, device).to(device)
criterion = nn.MSELoss()
#print(list(model.parameters()))
optimizer = optim.Adam(list(model.parameters()), lr=args.lr)


def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        loss = criterion(model(input), target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
	if iteration > 100:
		break 

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def test():
    avg_psnr = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = model(input)
            mse = criterion(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))


def checkpoint(epoch):
    model_out_path = args.model+"_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

for epoch in range(1, args.nEpochs + 1):
    train(epoch)
    test()
    checkpoint(epoch)
