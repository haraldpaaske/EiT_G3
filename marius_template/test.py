from model import kinematic_NN
import torch
from modules import kin_plot
model = kinematic_NN()


model.load_state_dict(torch.load('marius_template/models/large_2e-06_transform.pht'))
model.eval()



test = torch.Tensor([2, 0.5, 1, 0.8, 2, 1])
goal_pose = tuple(test[:3].tolist())


ans = model.pred(test)

kin_plot(ans, goal_pose)

