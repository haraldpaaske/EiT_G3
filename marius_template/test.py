from model import kinematic_NN
import torch
from modules import kin_plot, transform
import os
import numpy as np
import time
import pandas as pd
from torch import pi
neurons = 100
num_layers = 10
lr = 5e-04
data = '60k'
normalize=True

start_time = time.time()
model = kinematic_NN(neurons=neurons, num_layers=num_layers)


# model.load_state_dict(torch.load(f'marius_template/models/{num_layers}_{neurons}/{lr}_{data}_0.5_5.pht'))
# model.eval()
model.load_state_dict(torch.load(f'marius_template/models/{num_layers}_{neurons}/final.pht'))
model.eval()
#INPUTS___________________
x = -100
y = -50
z = -10
e1 = 0
e2 = pi
e3 = 0
#__________________________



test = torch.Tensor([x,y,z,e1,e2,e3])

goal_pose = tuple(test.tolist())

ans = model.pred(test)

end_time = time.time()
angles = [f"{num:.1f}" for num in torch.rad2deg(ans).tolist()]
print(f'Predicted angles: {angles}')
print(f'Runtime: {end_time-start_time:.4f}s')
l2_norm = torch.norm(test[:3]-transform(ans.unsqueeze(0))[0][:3], p=2)
print(f'Position error: {l2_norm:.3}cm')

ans_num = ans.numpy()


os.makedirs(f'marius_template/plots/{num_layers}_{neurons}', exist_ok=True)
name = f'marius_template/plots/{num_layers}_{neurons}/{lr}_{data}_0.1_3'

kin_plot(ans, goal_pose, name)
