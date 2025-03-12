from torch import nn
from base import BaseLoss
import torch
from torch import cos, sin, sqrt, atan2, ones_like, zeros, zeros_like

PI = 3.14159


class MSELoss(BaseLoss):
    def transform_output(self, theta):
        size = theta.shape[0]
        theta = theta.float()
        device = theta.device

        r_90 = torch.pi/2
        r_180 = torch.pi

        alpha = torch.tensor([r_90, 0, -r_90, r_90, -r_90, 0],
                             dtype=torch.float32, device=device)
        d = torch.tensor([-50, -130, 5.5, 0, 0, 0],
                         dtype=torch.float32, device=device)
        r = torch.tensor([104.5, 0, 0, 102.5, 0, 23],
                         dtype=torch.float32, device=device)

        theta = torch.column_stack([
            torch.full((size,), r_180, device=device) + theta[:, 0],
            torch.full((size,), r_90, device=device) + theta[:, 1],
            theta[:, 2],
            theta[:, 3],
            theta[:, 4],
            theta[:, 5],
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

        T = T.permute(2, 0, 1, 3)

        result = torch.eye(
            4, dtype=torch.float32, device=device).unsqueeze(-1).expand(4, 4, size).clone()
        result = result.permute(2, 0, 1)

        # Batch-wise matrix multiplication from T1....Tn
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

    def get_loss(self):
        return nn.MSELoss()
