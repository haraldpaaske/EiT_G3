import torch
from torch import cos, sin, atan2, sqrt, zeros, zeros_like, ones, ones_like, deg2rad, asin
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sm
import torch
import numpy as np

PI = 3.14159

def transform(theta):
    r_90 = torch.pi/2
    r_180 = torch.pi
    size = theta.shape[0]
    alpha = torch.tensor([r_90,0,-r_90,r_90,-r_90,0], device=theta.device)
    d= torch.tensor([-50, -130, 5.5, 0, 0, 0], device=theta.device)
    r = torch.tensor([104.5, 0, 0, 102.5, 0, 23], device=theta.device)

    theta = torch.column_stack([ 
                            torch.full((size,), r_180, device=theta.device) + theta[:, 0],
                            torch.full((size,), r_90, device=theta.device) + theta[:,1],
                            theta[:,2],
                            theta[:,3],
                            theta[:,4],
                            theta[:,5],
                            ])
    N = 6

    alpha = alpha.unsqueeze(0).expand(size,-1)
    r = r.unsqueeze(0).expand(size,-1)
    d = d.unsqueeze(0).expand(size,-1)

    T = torch.stack([
                    torch.stack([cos(theta), -sin(theta)*cos(alpha), sin(theta)*sin(alpha), r*cos(theta)]),
                    torch.stack([sin(theta), cos(theta)*cos(alpha), -cos(theta)*sin(alpha), r*sin(theta)]),
                    torch.stack([zeros_like(theta), sin(alpha), cos(alpha), d]),
                    torch.stack([zeros_like(theta), zeros_like(theta), zeros_like(theta), ones_like(theta)]),
                    ])

    T = T.permute(2,0,1,3)

    result = torch.eye(
        4, dtype=torch.float32, device=theta.device).unsqueeze(-1).expand(4, 4, size).clone()
    result = result.permute(2,0,1)


    #Batch-wise matrix multiplication from T1....Tn
    for i in range(N):
        result = torch.bmm(result, T[...,i])

    R = result[:,:3,:3]

    #Extract position and euler angles
    x = result[:,0,3]
    y = result[:,1,3]
    z = result[:,2,3]

    e1 = atan2(R[:,2,1],R[:,2,2])
    e2 = atan2(-R[:,2,0], sqrt(R[:,2,1]**2+R[:,2,2]**2))
    e3 = atan2(R[:,1,0], R[:,0,0])

    # e2 = asin(-R[:,2,0])

    output = torch.stack([x,y,z,e1,e2,e3], dim=1)
    return output


class DataFrameDataset(Dataset):
    def __init__(self, dataframe):
        self.data = torch.tensor(dataframe.iloc[:, :6].values, dtype=torch.float32)  # Features
        self.labels = torch.tensor(dataframe.iloc[:, 6:].values, dtype=torch.float32)  # Labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    

def kin_plot(theta, goal):
    theta = theta.detach().numpy()
    t_s, a_s, r_s, d_s = sm.symbols('θ α a d')

    T = sm.Matrix([[sm.cos(t_s), -sm.sin(t_s)*sm.cos(a_s),  sm.sin(t_s)*sm.sin(a_s), r_s*sm.cos(t_s)],
               [sm.sin(t_s),  sm.cos(t_s)*sm.cos(a_s), -sm.cos(t_s)*sm.sin(a_s), r_s*sm.sin(t_s)],
               [    0     ,          sm.sin(a_s)   ,           sm.cos(a_s)  ,        d_s        ],
               [    0     ,            0          ,                 0     ,        1        ]])

    params = sm.Matrix([t_s, a_s, r_s, d_s])
    T_i_i1 = sm.lambdify((params,), T, modules='numpy')

    #__________________________________________
    alpha = np.array([np.radians(90),0,np.radians(-90),np.radians(90),np.radians(-90),0])
    d= np.array([-50,-130,5.5,0,0,0,])
    r = np.array([104.5,0,0,102.5,0,23])


    theta = np.column_stack([ 
                            np.radians(180) + theta[0],
                            np.radians(90) + theta[1],
                            theta[2],
                            theta[3],
                            theta[4],
                            theta[5],
                            ])

    

    param = np.array([theta[0], alpha, r, d])
    param= np.transpose(param)
    print(param)

    points = np.array([[0,0,0]])
    Tt = np.eye(4)
    for par in param:
        Tt = Tt @ T_i_i1(par)
        points = np.vstack((points, Tt[:3,3]))


    X, Y, Z = points[:,0], points[:,1], points[:,2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X, Y, Z, '-o', markersize=8, label="Robot Arm")
    ax.scatter(X, Y, Z, color='r', s=50)  # Mark joints
    ax.scatter(goal[0], goal[1], goal[2], color='y', s=100)
    # Label axes
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("3D Robot Arm Visualization")
    ax.legend()
    plt.savefig('marius_template/test_plot/10_100_5e-06_10epochs.png')
    plt.show()

