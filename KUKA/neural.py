import torch 
import torch.nn as nn
import torch.nn.functional as F

class KinematicNN(nn.Module):
    def __init__(self, in_dim=6, out_dim=6):
        super(KinematicNN, self).__init__()
        self.input = nn.Linear(in_features=in_dim, out_features=20)
        self.hidden_1 = nn.Linear(in_features=20, out_features=20)
        self.hidden_2 = nn.Linear(in_features=20, out_features=20)
        self.output = nn.Linear(in_features=20, out_features=out_dim)
    
    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = self.output(x)
        return x
    
    def test(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)