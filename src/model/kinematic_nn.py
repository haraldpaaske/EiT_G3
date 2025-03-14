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


class BNKinematicNN(BaseModel):
    """
    Example: MLP with batch normalization
    """
    def __init__(self, neurons=100, num_layers=6, in_dim=6, out_dim=6):
        super().__init__(in_dim, out_dim)

        layers = []
        # Input
        layers.append(nn.Linear(in_dim, neurons))
        layers.append(nn.BatchNorm1d(neurons))
        layers.append(nn.ReLU())

        # Hidden
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(neurons, neurons))
            layers.append(nn.BatchNorm1d(neurons))
            layers.append(nn.ReLU())

        # Output
        layers.append(nn.Linear(neurons, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def test(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)


class LNKinematicNN(BaseModel):
    """
    Example: MLP with layer normalization
    """
    def __init__(self, neurons=100, num_layers=6, in_dim=6, out_dim=6):
        super().__init__(in_dim, out_dim)

        layers = []
        # Input
        layers.append(nn.Linear(in_dim, neurons))
        layers.append(nn.LayerNorm(neurons))
        layers.append(nn.ReLU())

        # Hidden
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(neurons, neurons))
            layers.append(nn.LayerNorm(neurons))
            layers.append(nn.ReLU())

        # Output
        layers.append(nn.Linear(neurons, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def test(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)
