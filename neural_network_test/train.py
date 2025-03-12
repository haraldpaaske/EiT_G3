import torch
import torch.nn as nn
from neural_network_test.neural_network import kinematic_NN_2
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from modules.mods import scara_forward_kinematics_2D

train = 'dataset/dataset1000_train.json'
df = pd.read_json(train)


class DataFrameDataset(Dataset):
    def __init__(self, dataframe):
        self.data = torch.tensor(dataframe.iloc[:, :2].values, dtype=torch.float32)  # Features
        self.labels = torch.tensor(dataframe.iloc[:, 2:].values, dtype=torch.float32)  # Labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]


train_set = DataFrameDataset(df)

dataloader = DataLoader(train_set, batch_size=4, shuffle=True)


model = kinematic_NN_2()
lr = 0.0005
num_epochs = 30

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

model.train()
loss_list = []
for epoch in range(num_epochs):
    running_loss = 0
    for i, batch in enumerate(dataloader):
        features, labels = batch
        optimizer.zero_grad()
        output = model(features)
        
        
        x = output[:,0]
        y = output[:,1]


        out_pos = scara_forward_kinematics_2D(x, y, 1.0, 0.8, arg='train')


        loss = criterion(out_pos, features)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        
    loss_list.append(running_loss)
        # print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss:.3f}')
        
            
torch.save(model.state_dict(), 'model_weights_big.pht')    



plt.figure(figsize=(8,5))
plt.plot(loss_list)
plt.savefig(f'loss_plot')






