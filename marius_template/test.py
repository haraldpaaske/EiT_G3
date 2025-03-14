from model import kinematic_NN
import torch
from modules import kin_plot, transform
neurons = 500
num_layers = 10
lr = 5e-4
data = '30k'

model = kinematic_NN(neurons=neurons, num_layers=num_layers)

model.load_state_dict(torch.load(f'marius_template/models/{num_layers}_{neurons}/{lr}_{data}.pht'))
model.eval()



test = torch.Tensor([185.4465151047, 148.2513909457, -106.8366286448, 1.592717156, 0.3575709642, -1.2239025072])
goal_pose = tuple([185.4465151047, 148.2513909457, -106.8366286448])



ans = model.pred(test)
print(transform(ans))

kin_plot(ans, goal_pose)


