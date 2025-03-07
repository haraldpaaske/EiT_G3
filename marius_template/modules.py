import torch
from torch import cos, sin, atan2, sqrt, zeros, zeros_like, ones, ones_like

def transform(theta):
    size = theta.shape[0]
    alpha = torch.Tensor([0,90,90,0,-90,-90,90,-90,0])
    d= torch.Tensor([0,0.479,0.5,0.178,0,0.0557,0.536,0,0.237])
    r = torch.Tensor([0.566,-0.067,0,1.3,0.489,0,0,0,0])

    theta = torch.column_stack([torch.zeros(size), 
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

    output = torch.stack([x,y,z,e1,e2,e3], dim=1)
    return output