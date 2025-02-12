import torch
import torch.nn as nn
from nn_test import kinematic_NN
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt



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


dataloader = DataLoader(train_set, batch_size=8, shuffle=True)


model = kinematic_NN()
lr = 0.001
num_epochs = 25

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
        
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        
        

        
    loss_list.append(running_loss)
        # print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss:.3f}')
        
            
    

plt.plot(loss_list)
plt.show()


torch.save(model.state_dict(), 'model_weights.pht')



