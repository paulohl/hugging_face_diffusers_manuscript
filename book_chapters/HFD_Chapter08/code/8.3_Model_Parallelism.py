# In this example, the model's layers are split across two GPUs, 
# and the data is transferred between devices during the forward pass. 
# This ensures that no single GPU is overloaded with memory-intensive tasks, 
# enabling the training of large models on hardware with limited capacity.

import torch 
from torch import nn c

lass Model(nn.Module): 
def __init__(self): 
     super(Model, self).__init__() 
     self.layer1 = nn.Linear(1000, 500).to('cuda:0') # Assign to GPU 0 
     self.layer2 = nn.Linear(500, 10).to('cuda:1') # Assign to GPU 1 

def forward(self, x): 
	x = self.layer1(x.to('cuda:0'))
           x = self.layer2(x.to('cuda:1')) 
           return x   

model = Model() input_data = torch.randn(64, 1000) output = model(input_data)
