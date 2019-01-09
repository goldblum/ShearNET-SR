import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np


arr = np.arange(12).reshape(3, 4) 
print("\n\n2D arr : \n", arr) 
print("Shape : ", arr.shape) 
  
a = np.insert(arr, 1, 9, axis = 1) 
print("\nArray after insertion : \n", a) 
print("Shape : ", a.shape) 


x = np.array( [[1,0],[0,1]] )
b = np.array([0,0])
x = np.insert(x, 1, b, axis = 1)
print(x)

'''
C = nn.Parameter(torch.randn(32, 64, 3, 3))
A = np.array( [[1,0,0,0,0], [0,0,1,0,0], [0,0,0,0,1]] )

A = torch.from_numpy(A)
A = A.type(torch.FloatTensor)
print((torch.matmul(torch.t(A),(torch.matmul(C, A))).shape))

C = np.matmul(A,B)
print(C)
D = np.matmul(np.transpose(B), C)
print(D)
'''

