import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

model = torch.load("model_epoch_1.pth")
kernel = (model.lay1.conv_weight).detach().numpy()
print(np.shape(kernel))
#fig = plt.figure()
#plt.subplot
#plt.imshow(kernel[0][0,:,:], cmap = 'Greys')
#plt.show()
