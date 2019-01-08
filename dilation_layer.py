import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):

    def __init__(self, depth, input_depth):
        super(BasicBlock, self).__init__()
	self.conv_weight = nn.Parameter(torch.randn(depth, input_depth,3, 3))
        self.conv1 = F.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
        self.conv2 = F.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=2, groups=1)
      

    def forward(self, x):
	x_depth = (list(x.shape))[2]
        out1 = F.relu(self.conv1(x))
        out2 = F.relu(self.conv2(x))
        return torch.cat(out1,out2)
