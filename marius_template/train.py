from modules import transform, DataFrameDataset
import torch
import torch.nn as nn
from model import kinematic_NN
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler

neurons = 100
num_layers = 10
lr = 5e-4
num_epochs = 50
normalize = True
sched = True
cos_dec = False
gamma = 0.6
step_size = 5


samp = '60k'
train = 'KUKA/data/dataset/dataset60000/train.json'
val = 'KUKA/data/dataset/dataset60000/val.json'
train_df = pd.read_json(train)
val_df = pd.read_json(val)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print(f"Using device: {device}")

#______NORMALIZER_________

scaler = MinMaxScaler(feature_range=(-1,1))
if normalize:
    train_df = pd.DataFrame(scaler.fit_transform(train_df), columns=train_df.columns)
    val_df = pd.DataFrame(scaler.transform(val_df), columns=val_df.columns)

    x_min = torch.Tensor(scaler.data_min_)[:6]
    x_max = torch.Tensor(scaler.data_max_)[:6]
    x_min = x_min.to(device)
    x_max = x_max.to(device)

    

def de_normalize(pred):
    return (pred+torch.ones_like(pred)) / 2 * (x_max-x_min)+x_min
   
train_set = DataFrameDataset(train_df)
val_set = DataFrameDataset(val_df)
dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
valloader = DataLoader(val_set, batch_size=1)

model = kinematic_NN(num_layers=num_layers, neurons=neurons)
model = model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# lambda1 = lambda epoch: 0.2**epoch
if sched:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=gamma, step_size=step_size)

elif cos_dec:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr*0.001)

loss_list = []
val_list = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for i, batch in enumerate(dataloader):
        features, labels = batch
        features, labels = features.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        output = model(features)
        if normalize:
            output = de_normalize(output)
            features = de_normalize(features)

        out_pos = transform(output)
        loss = criterion(out_pos, features)
        loss.backward()
        optimizer.step()
        
        running_loss+=loss.item()
    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss:.3f}')
    if sched:
        scheduler.step()

    #______VALIDATE__________
    model.eval()
    with torch.no_grad():
        l2 = []
        for features, _ in valloader:
            features = features.to(device, non_blocking=True)     
            output = model(features)
            if normalize:
                output = de_normalize(output)
                features = de_normalize(features)
            predicted_pos = transform(output)
            
            l2_norm = torch.norm(features-predicted_pos, p=2)
            l2.append(l2_norm)

        score = sum(l2)/len(l2)
        print(f'[{epoch + 1}, {i + 1:5d}] L2-score: {score:.3f}')
        val_list.append(score.cpu().item())
    loss_list.append(running_loss)
    
os.makedirs(f'marius_template/models/{num_layers}_{neurons}', exist_ok=True)          
torch.save(model.state_dict(), f'marius_template/models/{num_layers}_{neurons}/{lr}_{samp}_{gamma}_{step_size}_norm_tanh.pht')    



#____________________Loss and validation plot_______________________________
fig, axes = plt.subplots(1, 2, figsize=(16, 5))  # 1 row, 2 columns

axes[0].plot(loss_list)
axes[0].set_title(f'lr = {lr} - Loss')
axes[0].grid(True)
axes[0].set_ylim([0, 1000000])
axes[1].plot(val_list)
axes[1].set_title(f'lr = {lr} - Validation')
axes[1].grid(True)
plt.tight_layout()

# Save the combined figure
os.makedirs(f'marius_template/loss/{num_layers}_{neurons}', exist_ok=True)
plt.savefig(f'marius_template/loss/{num_layers}_{neurons}/{lr}_{samp}_{gamma}_{step_size}_norm_tanh.png')


