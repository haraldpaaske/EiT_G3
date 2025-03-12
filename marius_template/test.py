from model import kinematic_NN
import torch
from modules import kin_plot, transform
neurons = 100
num_layers = 10
lr = 5e-06
data = '30k'

model = kinematic_NN(neurons=neurons, num_layers=num_layers)

model.load_state_dict(torch.load(f'marius_template/models/{num_layers}_{neurons}/{lr}_{data}_new.pht'))
model.eval()



test = torch.Tensor([-100, -50, 80, 0.8, 2, 1])
goal_pose = tuple(test[:3].tolist())

#[-107.8429,  -75.8332,   39.3367,   -3.0498,   -0.7225,   -0.9642]

ans = model.pred(test)



kin_plot(ans, goal_pose)


