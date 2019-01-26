import torch
import torch.nn as nn
import math

class _Residual_Block(nn.Module):
    def __init__(self, res_trainable_channels):
        super(_Residual_Block, self).__init__()
	self.res_trainable_channels = res_trainable_channels
        self.conv1 = nn.Conv2d(in_channels=self.res_trainable_channels, out_channels=self.res_trainable_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(self.res_trainable_channels, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=self.res_trainable_channels, out_channels=self.res_trainable_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(self.res_trainable_channels, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output,identity_data)
        return output 

class Net(nn.Module):
    def __init__(self, device, res_trainable_channels = 4, im_size = 14):
        super(Net, self).__init__()
	self.res_trainable_channels = res_trainable_channels
        self.conv_input = nn.Conv2d(in_channels=4, out_channels = self.res_trainable_channels , kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.residual = self.make_layer(_Residual_Block, 16)
        self.conv_mid = nn.Conv2d(in_channels=self.res_trainable_channels, out_channels=self.res_trainable_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(self.res_trainable_channels, affine=True)
        self.upscale2x = nn.Sequential(
            nn.Conv2d(in_channels=self.res_trainable_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_output = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=9, stride=1, padding=4, bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(self.res_trainable_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out,residual)
        out = self.upscale2x(out)
        out = self.conv_output(out)
        return out


def SRResNet(device, trainable_channels):
	model = Net(device, trainable_channels)
	return model
