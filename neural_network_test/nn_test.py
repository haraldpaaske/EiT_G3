import torch 
import torch.nn as nn
import torch.nn.functional as F



class kinematic_NN(nn.Module):
    def __init__(self, in_dim=2, out_dim=2):
        super(kinematic_NN, self).__init__()
        self.input = nn.Linear(in_features= in_dim, out_features=5)
        self.hidden_1 = nn.Linear(in_features= 5, out_features=5)
        self.hidden_2 = nn.Linear(in_features= 5, out_features=5)
        # self.hidden_3 = nn.Linear(in_features= 20, out_features=20)
        # self.hidden_4 = nn.Linear(in_features= 20, out_features=20)
        self.output = nn.Linear(in_features=5, out_features=out_dim)
        pass
    
    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        # x = F.relu(self.hidden_3(x))
        # x = F.relu(self.hidden_4(x))
        x = self.output(x)
        return x
    

