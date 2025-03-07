from forward_kin_torch import forward_6dof
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import sympy as sm
import torch
from torch import cos,sin, sqrt, atan2, zeros_like, zeros, ones_like
import sys

output = torch.tensor([
    [ 109.7076815291, -83.3482700545,  -51.6722432475,  -307.3443919315,  120.0, -16.571066378],
    [ -176.4902101058, -122.0065670151,  -62.6233553482,  -25.5873180229,  120.0, -131.2468911226]
], requires_grad=True)


α_i = [0,90,90,0,-90,-90,90,-90,0]
d_i= [0,0.479,0.5,0.178,0,0.0557,0.536,0,0.237]
r_i = [0.566,-0.067,0,1.3,0.489,0,0,0,0]

out=[]

α_i = [0,90,90,0,-90,-90,90,-90,0]
α = [torch.tensor([x]) for x in α_i]
d_i= [0,0.479,0.5,0.178,0,0.0557,0.536,0,0.237]
d = [torch.tensor([x]) for x in d_i]
r_i = [0.566,-0.067,0,1.3,0.489,0,0,0,0]
r = [torch.tensor([x]) for x in r_i]

for j in range(output.shape[0]): #number of batches
    th = output[j]               #j'th batch
    
    T = []
    t1 = torch.Tensor([0])
    t2 = torch.Tensor([90]) + th[0]
    t3 = torch.Tensor([90])
    t4 = th[1]
    t5 = torch.Tensor([90]) + th[2]
    t6 = torch.Tensor([-90])
    t7 = torch.Tensor([90]) + th[3]
    t8= th[4]
    t9 = th[5]
    t = torch.column_stack((t1, t2, t3, t4, t5, t6, t7, t8, t9))
    t=t[0]
    # print(f't: {t}')
    for i in range(9):
        thetaa = t[i:(i+1)]
        
        Tt = torch.Tensor([[ cos(t[i:(i+1)]), -sin(t[i:(i+1)])*cos(t[i:(i+1)]),  sin(t[i:(i+1)])*sin(α[i])  ,  r[i]*cos(t[i:(i+1)]) ],
                            [sin(t[i:(i+1)]),  cos(t[i:(i+1)])*cos(α[i]), -cos(t[i:(i+1)])*sin(α[i]) ,  r[i]*sin(t[i:(i+1)])  ],
                            [    0     ,          sin(α[i])   ,           cos(α[i]),      d[i]        ],
                            [    0     ,            0          ,              0    , torch.Tensor([1])]
        #                     ])
        # Tt = torch.stack([
        #         torch.stack([torch.cos(thetaa), -torch.sin(thetaa) * torch.cos(α[i]),  torch.sin(thetaa) * torch.sin(α[i]),  r[i] * torch.cos(thetaa)]),
        #         torch.stack([torch.sin(t[i:(i+1)]),  torch.cos(t[i:(i+1)]) * torch.cos(α[i]), -torch.cos(t[i:(i+1)]) * torch.sin(α[i]),  r[i] * torch.sin(thetaa)]),
        #         torch.stack([torch.tensor(0.0, device=t.device), torch.sin(α[i]), torch.cos(α[i]), d[i]]),
        #         torch.stack([torch.tensor(0.0, device=t.device), torch.tensor(0.0, device=t.device), torch.tensor(0.0, device=t.device), torch.tensor(1.0, device=t.device)])
])

        print(f'Tt_{i}: {Tt}')
        T.append(Tt)

    T_1_6 = T[0]

    for mat in T[1:]:
        T_1_6 = torch.matmul(T_1_6, mat)

    # T_1_6 = T[0]@T[1]@T[2]@T[3]@T[4]@T[5]@T[6]@T[7]@T[8]

    R = T_1_6[:3, :3]     #Rotation matrix from base to end effector
    tr = T_1_6[:3, 3]      #Translation from base to end effector
    x = tr[0]
    y = tr[1]
    z = tr[2]

    phi = atan2(R[2,1],R[2,2])
    thh = atan2(-R[2,0], sqrt(R[2,1]**2+R[2,2]**2))
    psi = atan2(R[1,0], R[0,0])
    out.append(torch.Tensor([x,y,z, phi, thh, psi]))
    
print(out)