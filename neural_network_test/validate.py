import torch
import torch.nn as nn
from neural_network_test.kinematic_nn import kinematic_NN
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from modules.mods import scara_forward_kinematics_2D

model = kinematic_NN()

model.load_state_dict(torch.load('model_weights_working.pht'))
model.eval()
# test = torch.tensor([[1.6, 0.5]])

# ans = model.forward(test)

# t1 = ans[0][0].item()
# t2 = ans[0][1].item()


test = 'dataset/dataset1000_test.json'
df = pd.read_json(test)
class DataFrameDataset(Dataset):
    def __init__(self, dataframe):
        self.data = torch.tensor(dataframe.iloc[:, :2].values, dtype=torch.float32)  # Features
        self.labels = torch.tensor(dataframe.iloc[:, 2:].values, dtype=torch.float32)  # Labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]

test_set = DataFrameDataset(df)
testloader = DataLoader(test_set, batch_size=1, shuffle=True)


model.eval()
t1_loss = 0
t2_loss = 0
for i,data in enumerate(testloader):
    features, labels = data

    ans = model.forward(features)

    t1 = ans[0][0].item()
    t2 = ans[0][1].item()

    t1_true, t2_true = labels[0][0].item(), labels[0][1].item()

    x,y = scara_forward_kinematics_2D(t1, t2, L1=1.0, L2=0.8, arg='dataset')

    t1_loss += (features[0]-t1)**2+(features[1]-t2)**2  


    
avg_t1 = t1_loss/i


print(f'theta 1 MSE: {avg_t1}')




# def scara_forward_kinematics(theta1, theta2, L1=1, L2=0.8):
#     """
#     Computes the end-effector position (x, y, z) and orientation theta_end.

#     Parameters:
#     theta1, theta2, theta3 (float): Joint angles in radians.
#     d (float): Vertical displacement (prismatic joint).
#     L1, L2 (float): Length of the first and second links.

#     Returns:
#     tuple: (x, y, z, theta_end)
#     """
#     x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
#     y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)

#     return x, y