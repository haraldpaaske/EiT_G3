import torch
import torch.nn as nn
from base import BaseModel

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # One simple approach is: x -> linear1 -> ReLU -> linear2 -> ReLU
        # and add it back to the original x
        identity = x
        out = self.relu(self.linear1(x))
        out = self.relu(self.linear2(out))
        return out + identity


class ResidualKinematicNN(BaseModel):
    """
    Example: a deeper network with residual blocks
    """
    def __init__(self, in_dim=6, out_dim=6, neurons=256, num_blocks=4):
        super().__init__(in_dim, out_dim)

        self.input_layer = nn.Linear(in_dim, neurons)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(neurons) for _ in range(num_blocks)
        ])
        self.output_layer = nn.Linear(neurons, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        for block in self.res_blocks:
            x = block(x)
        return self.output_layer(x)

    def test(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)
