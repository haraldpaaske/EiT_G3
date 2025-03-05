import numpy as np
import torch

def scara_forward_kinematics_2D(theta1, theta2, L1, L2, arg):
    """
    Computes the end-effector position (x, y, z) and orientation theta_end.

    Parameters:
    theta1, theta2, theta3 (float): Joint angles in radians.
    d (float): Vertical displacement (prismatic joint).
    L1, L2 (float): Length of the first and second links.

    Returns:
    tuple: (x, y, z, theta_end)
    """
    x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
    
    if arg == 'train':
        return torch.stack((x,y), axis=1)

    if arg == 'dataset':
        return x, y
    return x, y

