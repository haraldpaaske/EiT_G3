from model import kinematic_NN
import torch
from modules import kin_plot
neurons = 10
num_layers = 10
lr = 5e-06
data = '30k'

model = kinematic_NN(neurons=neurons, num_layers=num_layers)

model.load_state_dict(torch.load(f'marius_template/models/{num_layers}_{neurons}/{lr}_{data}.pht'))
model.eval()



test = torch.Tensor([-100, 50, 80, 0.8, 2, 1])
goal_pose = tuple(test[:3].tolist())


ans = model.pred(test)

kin_plot(ans, goal_pose)

