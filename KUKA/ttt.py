from forward_kin_torch import forward_6dof
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import sympy as sm
import torch
from torch import cos,sin, sqrt, atan2, zeros_like, zeros, ones_like
import sys

theta = torch.tensor([
    [ 109.7076815291, -83.3482700545,  -51.6722432475,  -307.3443919315,  120.0, -16.571066378],
    [ -176.4902101058, -122.0065670151,  -62.6233553482,  -25.5873180229,  120.0, -131.2468911226]
], requires_grad=True)


size = theta.shape[0]
alpha = torch.Tensor([0,90,90,0,-90,-90,90,-90,0])
d= torch.Tensor([0,0.479,0.5,0.178,0,0.0557,0.536,0,0.237])
r = torch.Tensor([0.566,-0.067,0,1.3,0.489,0,0,0,0])

Theta = torch.column_stack([torch.zeros(size), 
                        torch.full((size,), 90) + theta[:,0],
                        torch.full((size,), 90),
                        theta[:,1],
                        torch.full((size,), 90) + theta[:,2],
                        torch.full((size,), -90),
                        torch.full((size,), 90) + theta[:,3],
                        theta[:,4],
                        theta[:,5],
                        ])
N = 9
theta = Theta

alpha = alpha.unsqueeze(0).expand(size,-1)
r = r.unsqueeze(0).expand(size,-1)
d = d.unsqueeze(0).expand(size,-1)

T = torch.stack([
                torch.stack([cos(theta), -sin(theta)*cos(theta), sin(theta)*sin(alpha), r*cos(theta)]),
                torch.stack([sin(theta), cos(theta)*cos(alpha), -cos(theta)*sin(alpha), r*sin(theta)]),
                torch.stack([zeros_like(theta), sin(alpha), cos(alpha), d]),
                torch.stack([zeros_like(theta), zeros_like(theta), zeros_like(theta), ones_like(theta)]),
                ])

T = T.permute(2,0,1,3)

result = torch.eye(4).unsqueeze(-1).expand(4, 4, size).clone()
result = result.permute(2,0,1)


for i in range(N):
    result = torch.bmm(result, T[...,i])
    




