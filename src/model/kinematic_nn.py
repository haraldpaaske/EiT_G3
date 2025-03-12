import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class KinematicNN(BaseModel):

    def __init__(self, neurons=100, num_layers=6, in_dim=6, out_dim=6):
        super(KinematicNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, neurons))
        for _ in range(num_layers):
            self.layers.append(nn.Linear(neurons, neurons))
        
        self.layers.append(nn.Linear(neurons, out_dim))
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = nn.ReLU()(layer(x))

        return self.layers[-1](x)
    
    def test(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)
