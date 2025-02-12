import torch
import torch.nn as nn
from nn_test import kinematic_NN
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

model = kinematic_NN()

model.load_state_dict(torch.load('model_weights.pht'))
model.eval()
# test = torch.tensor([[1.6, 0.5]])

# ans = model.forward(test)

# t1 = ans[0][0].item()
# t2 = ans[0][1].item()


def scara_forward_kinematics(theta1, theta2, L1=1, L2=0.8):
    """
    Computes the end-effector position (x, y, z) and orientation theta_end.

    Parameters:
    theta1, theta2, theta3 (float): Joint angles in radians.
    d (float): Vertical displacement (prismatic joint).
    L1, L2 (float): Length of the first and second links.

    Returns:
    tuple: (x, y, z, theta_end)
    """
    x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)

    return x, y


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

    t1_loss += (t1_true-t1)**2
    t2_loss += (t2_true-t2)**2

    
avg_t1 = t1_loss/i
avg_t2 = t2_loss/i

print(f'theta 1 MSE: {avg_t1}')
print(f'theta 2 MSE: {avg_t2}')
