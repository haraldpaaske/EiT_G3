import torch
import torch.nn as nn
from base import BaseModel

class MultiHeadKinematicNN(BaseModel):
    """
    Branches into a 'position head' and an 'orientation head'.
    out_dim = 6 -> 3 for position, 3 for orientation
    """
    def __init__(self, in_dim=6, neurons=256, num_layers=4):
        super().__init__(in_dim, out_dim=6)
        self.shared_layers = nn.ModuleList()
        self.shared_layers.append(nn.Linear(in_dim, neurons))
        for _ in range(num_layers - 1):
            self.shared_layers.append(nn.Linear(neurons, neurons))

        self.pos_head = nn.Linear(neurons, 3)
        self.ori_head = nn.Linear(neurons, 3)

        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.shared_layers:
            x = self.relu(layer(x))

        pos = self.pos_head(x)
        ori = self.ori_head(x)

        return torch.cat([pos, ori], dim=-1)

    def test(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)
