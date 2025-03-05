from forward_kin_torch import forward_6dof
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import sympy as sm
import torch
from torch import cos,sin, sqrt, atan2
num_samples = 1

α_i = [0,90,90,0,-90,-90,90,-90,0]
d_i= [0,0.479,0.5,0.178,0,0.0557,0.536,0,0.237]
r_i = [0.566,-0.067,0,1.3,0.489,0,0,0,0]

theta = torch.tensor([
    [ 78.2489474053, -70.6465328702,  89.4170801733,  -335.7446323521,  120.0, -79.1094719952],
    [ 109.7076815291, -83.3482700545,  -51.6722432475,  -307.3443919315,  120.0, -16.571066378],
    [ -2.8391, -84.8621,  40.1715, -11.6187, 188.1395, -23.3062],
    [ -2.2983, -40.7658,  19.1716,  -8.0582,  91.9025, -11.3721],
    [ -0.9639, -38.6740,  17.6525,  -6.9980,  86.2602,  -9.5502],
    [ -0.3806, -42.8694,  19.9867,  -9.2966,  96.8384,  -9.1422],
    [ -0.6030, -70.3727,  33.6428,  -7.5096, 158.1611, -23.4686],
    [ -0.6076, -35.6111,  16.0049,  -7.4067,  79.8239,  -8.1022]
], requires_grad=True)


def forward_kin(theta):
    out=[]
    # n = theta.shape[1]

    α_i = [0,90,90,0,-90,-90,90,-90,0]
    α = [torch.tensor([x]) for x in α_i]
    d_i= [0,0.479,0.5,0.178,0,0.0557,0.536,0,0.237]
    d = [torch.tensor([x]) for x in d_i]
    r_i = [0.566,-0.067,0,1.3,0.489,0,0,0,0]
    r = [torch.tensor([x]) for x in r_i]
    
    for j in range(theta.shape[0]): #number of batches
        th = theta[j]               #j'th batch
        
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
        for i in range(9):
            
            Tt = torch.Tensor([[cos(t[i]), -sin(t[i])*cos(t[i]),  sin(t[i])*sin(α[i]), r[i]*cos(t[i])],
               [sin(t[i]),  cos(t[i])*cos(α[i]), -cos(t[i])*sin(α[i]), r[i]*sin(t[i])],
               [    0     ,          sin(α[i])   ,           cos(α[i])  ,        d[i]        ],
               [    0     ,            0          ,                 0     ,        torch.Tensor([1])        ]])
            
            T.append(Tt)

        T_1_6 = T[0]@T[1]@T[2]@T[3]@T[4]@T[5]@T[6]@T[7]@T[8]

        R = T_1_6[:3, :3]     #Rotation matrix from base to end effector
        tr = T_1_6[:3, 3]      #Translation from base to end effector

        x = tr[0]
        y = tr[1]
        z = tr[2]

        phi = atan2(R[2,1],R[2,2])
        thh = atan2(-R[2,0], sqrt(R[2,1]**2+R[2,2]**2))
        psi = atan2(R[1,0], R[0,0])
        out.append(torch.Tensor([x,y,z, phi, thh, psi]))
        
    return torch.stack(out)


test = forward_kin(theta)

print(test[1])



