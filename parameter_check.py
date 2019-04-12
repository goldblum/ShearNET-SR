import torch
import torch.nn as nn
from model import Net
import Models
import numpy as np


#load_path1 = './ShearAndDilate_ch=4_1x_blah/checkpoints/epoch_1=.pth'
load_path1 = './ResNet_ch=4/checkpoints/epoch_200=.pth'
#load_path1 = './epoch_114=.pth'
#res_channels1 = 4
#model_function = 'SRResNet_shearAndDilate_3d'
device = 'cuda'

#modelFns = {'SRResNet':Models.SRResNet.SRResNet, 'SRResNet_shear':Models.SRResNet_shear.SRResNet_shear, 'SRResNet_dilate':Models.SRResNet_dilate.SRResNet_dilate, 'SRResNet_shearAndDilate':Models.SRResNet_shearAndDilate.SRResNet_shearAndDilate, 'SRResNet_shearAndDilate_3d':Models.SRResNet_shearAndDilate_3d.SRResNet_shearAndDilate_3d}

#modelFN = modelFns[ model_function ]

model = torch.load(load_path1, map_location=device)
#model = modelFN(device, res_channels1).to(device)
#print(checkpoint['model'])
#model.load_state_dict(checkpoint['model'])

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

for name, param in model.named_parameters():
    if param.requires_grad:
        print name, param.data.size()

print(pytorch_total_params)

#python main.py --nEpochs 200 --cuda --model "SRResNet_shearAndDilate_3d" --res_channels 32 --model_save "./ShearAndDilate_ch=32_3d" 



'''
3d:
15464
residual.0.conv1.conv_weight (4, 1, 1, 3, 3)
residual.0.conv1.conv1x1_weight (4, 40, 1, 1)
residual.0.in1.weight (4,)
residual.0.in1.bias (4,)
residual.0.conv2.conv_weight (4, 1, 1, 3, 3)
residual.0.conv2.conv1x1_weight (4, 40, 1, 1)
residual.0.in2.weight (4,)
residual.0.in2.bias (4,)


Regular ShearNet:
18920
residual.0.conv1.conv_weight (4, 4, 3, 3)
residual.0.conv1.conv1x1_weight (4, 40, 1, 1)
residual.0.in1.weight (4,)
residual.0.in1.bias (4,)
residual.0.conv2.conv_weight (4, 4, 3, 3)
residual.0.conv2.conv1x1_weight (4, 40, 1, 1)
residual.0.in2.weight (4,)
residual.0.in2.bias (4,)

SRResNet:
13800
residual.0.conv1.weight (4, 4, 3, 3)
residual.0.in1.weight (4,)
residual.0.in1.bias (4,)
residual.0.conv2.weight (4, 4, 3, 3)
residual.0.in2.weight (4,)
residual.0.in2.bias (4,)


'''

