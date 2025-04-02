from model import kinematic_NN
import torch
from modules import kin_plot, transform
import os
import numpy as np
neurons = 100
num_layers = 10
lr = 5e-04
data = '30k'

model = kinematic_NN(neurons=neurons, num_layers=num_layers)

model.load_state_dict(torch.load(f'marius_template/models/{num_layers}_{neurons}/{lr}_{data}_0.1_3.pht'))
model.eval()


test = torch.Tensor([-100.4465151047, -70.2513909457, -106.8366286448, 1.592717156, 0.3575709642, -1.2239025072])
goal_pose = tuple(test.tolist())

ans = model.pred(test)


ans_num = ans.numpy()
# print(np.rad2deg(ans_num))

os.makedirs(f'marius_template/plots/{num_layers}_{neurons}', exist_ok=True)
name = f'marius_template/plots/{num_layers}_{neurons}/{lr}_{data}_0.1_3'

kin_plot(ans, goal_pose, name)


