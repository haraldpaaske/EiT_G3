import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from neural_network_test.nn_test_big import kinematic_NN_2
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_robot_arm(theta1, theta2_relative, name, L1=1.0, L2=0.8):
    """
    Plots a 2-link robot arm where the second angle is relative to the first link.
    
    Parameters:
        theta1 (float): Absolute angle of the first joint (degrees)
        theta2_relative (float): Relative angle of the second joint (degrees)
        L1 (float): Length of the first link
        L2 (float): Length of the second link
    """

    # Convert angles to radians
    # theta1 = np.radians(theta1)  # Angle from x-axis
    # theta2 = np.radians(theta2_relative)  # Relative angle

    # Compute joint positions
    x0, y0 = 0, 0  # Base of the arm
    x1 = L1 * np.cos(theta1)
    y1 = L1 * np.sin(theta1)

    # Compute second joint angle as absolute (cumulative)
    theta2_absolute = theta1 + theta2_relative  # Now it's relative to the first link

    x2 = x1 + L2 * np.cos(theta2_absolute)
    y2 = y1 + L2 * np.sin(theta2_absolute)

    # Plot the arm
    plt.figure(figsize=(5, 5))
    plt.plot([x0, x1], [y0, y1], 'bo-', label="Link 1")  # First link
    plt.plot([x1, x2], [y1, y2], 'ro-', label="Link 2")  # Second link
    plt.scatter(name[0], name[1])
    # Plot joint locations
    plt.scatter([x0, x1, x2], [y0, y1, y2], c='black', zorder=3)  # Joints

    # Set plot limits
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.title(f"2-Link Robot Arm (θ1={theta1:.1f}°, θ2_relative={theta2_relative:.1f}°)")
    plt.xlabel("X")
    plt.ylabel("Y")
    # plt.xlim(-1.2,1.2)
    # plt.ylim(-1.2,1.2)
    plt.savefig(f'results/4_hidden_20_neurons/end_effector_{name}.png')

# Example usage



model = kinematic_NN_2()

model.load_state_dict(torch.load('results/4_hidden_20_neurons/model_weights_big.pht'))
model.eval()

input = torch.Tensor([[-1,1.5], [1,1], [1,-1.5], [-0.5,-0.5], [0.5,0.5]])


names = [(-1,1.5), (1,1), (1,-1.5), (-0.5,-0.5), (0.5,0.5)]


output = model(input)

x = output[:,0].detach().numpy()
y = output[:,1].detach().numpy()

print(x)

# theta1 = x[0]
# theta2_relative = y[0]

for theta1, theta2_relative, name in zip(x,y, names):

# print(x,y)

    plot_robot_arm(theta1, theta2_relative, name)
