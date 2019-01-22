import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
#import matplotlib.pyplot as plt

model = torch.load("SRResNet_shearAndDilate_epoch_1.pth")
#print(model.lay1.shear_layer())
kernel = (model.lay1.conv_weight).detach().numpy()
print(kernel)
#fig = plt.figure()
#plt.subplot
#plt.imshow(kernel[0][0,:,:], cmap = 'Greys')
#plt.show()
