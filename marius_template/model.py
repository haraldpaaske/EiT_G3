import torch 
import torch.nn as nn
import torch.nn.functional as F


class kinematic_NN(nn.Module):
    '''
    Neural network module
    Args:
        in_dim: Number of features in input data (x,y,z cordinates and θ,φ,ψ orientation)
        out_dim: Number of predictions (joint angles)
    '''

    def __init__(self, neurons=100, num_layers=6, in_dim=6, out_dim=6):
        super(kinematic_NN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, neurons)) #input layer

        for _ in range(num_layers):
            self.layers.append(nn.Linear(neurons, neurons)) #hidden layers

        self.layers.append(nn.Linear(neurons, out_dim)) #output layer
        # self.input = nn.Linear(in_features= in_dim, out_features=100)
        # self.hidden_1 = nn.Linear(in_features= 100, out_features=100)
        # self.hidden_2 = nn.Linear(in_features= 100, out_features=100)
        # self.hidden_3 = nn.Linear(in_features= 100, out_features=100)
        # self.hidden_4 = nn.Linear(in_features= 100, out_features=100)
        # self.hidden_5 = nn.Linear(in_features= 100, out_features=100)
        # self.hidden_6 = nn.Linear(in_features= 100, out_features=100)
        
        # self.output = nn.Linear(in_features=100, out_features=out_dim)
        pass
    
    def forward(self, x):
        '''
        Feedforward function for the neural network, used both for training and prediction. 
        Args:
            input: Torch tensor of size [in_dim, N]
        
        returns torch tensor of size [out_dim, N]

        (N: arbitrary sized dimention)
        ''' 
        for layer in self.layers[:-1]: #Relu on all but the output layer
            x = nn.ReLU()(layer(x))

        return self.layers[-1](x) #no activation on output layer
        # x = F.relu(self.input(x))
        # x = F.relu(self.hidden_1(x))
        # x = F.relu(self.hidden_2(x))
        # x = F.relu(self.hidden_3(x))
        # x = F.relu(self.hidden_4(x))
        # x = F.relu(self.hidden_5(x))
        # x = F.relu(self.hidden_6(x))
        # x = self.output(x)
        # return x
    
    def pred(self, x):
        with torch.no_grad():
        
            return self.forward(x)
