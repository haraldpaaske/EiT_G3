from model import kinematic_NN
import torch
from modules import kin_plot, transform
import os
neurons = 100
num_layers = 10
lr = 5e-04
data = '30k'

model = kinematic_NN(neurons=neurons, num_layers=num_layers)

model.load_state_dict(torch.load(f'marius_template/models/{num_layers}_{neurons}/{lr}_{data}_new.pht'))
model.eval()



test = torch.Tensor([185.4465151047, 148.2513909457, -106.8366286448, 1.592717156, 0.3575709642, -1.2239025072])
goal_pose = tuple(test[:3].tolist())



ans = model.pred(test)


os.makedirs(f'marius_template/plots/{num_layers}_{neurons}', exist_ok=True)
name = f'marius_template/plots/{num_layers}_{neurons}/{lr}_{data}'

kin_plot(ans, goal_pose, name)


