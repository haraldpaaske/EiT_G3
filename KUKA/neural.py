import torch 
import torch.nn as nn
import torch.nn.functional as F



class kinematic_NN_2(nn.Module):
    '''
    Neural network module
    Args:
        in_dim: Number of features in input data (x,y,z cordinates and θ,φ,ψ orientation)
        out_dim: Number of predictions (joint angles)
    '''

    def __init__(self, in_dim=6, out_dim=6):
        super(kinematic_NN_2, self).__init__()
        self.input = nn.Linear(in_features= in_dim, out_features=20)
        self.hidden_1 = nn.Linear(in_features= 20, out_features=20)
        self.hidden_2 = nn.Linear(in_features= 20, out_features=20)
        self.hidden_3 = nn.Linear(in_features= 20, out_features=20)
        self.hidden_4 = nn.Linear(in_features= 20, out_features=20)
        self.output = nn.Linear(in_features=20, out_features=out_dim)
        pass
    
    def forward(self, x):
        '''
        Feedforward function for the neural network, used both for training and prediction. 
        Args:
            input: Torch tensor of size [in_dim, N]
        
        returns torch tensor of size [out_dim, N]

        (N: arbitrary sized dimention)
        '''
        x = F.relu(self.input(x))
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = F.relu(self.hidden_3(x))
        x = F.relu(self.hidden_4(x))
        x = self.output(x)
        return x