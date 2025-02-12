import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# SCARA Robot Parameters (Modify Based on Your Setup)
L1 = 1  # Length of the first link
L2 = 0.8  # Length of the second link
d_min, d_max = 0.0, 0.0  # Limits for vertical displacement (prismatic joint)

# Joint Limits (Adjust for Your SCARA Robot)
theta1_min, theta1_max = np.radians(-90), np.radians(90)  # Base rotation
theta2_min, theta2_max = np.radians(-90), np.radians(90)  # Elbow rotation
# theta3_min, theta3_max = np.radians(-180), np.radians(180)  # End-effector rotation

# Number of Samples

num_samples = 1000

L1 = np.full(1000,1.0)
L2 = np.full(1000,0.8)

# Function for Forward Kinematics (FK)
def scara_forward_kinematics(theta1, theta2, L1, L2):
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
    
    return x, y

# Generate Dataset
dataset = []


theta1 = np.random.uniform(theta1_min, theta1_max, num_samples)
theta2 = np.random.uniform(theta2_min, theta2_max, num_samples)

x, y = scara_forward_kinematics(theta1, theta2, L1, L2)

dataset = np.array([[x], [y], [theta1], [theta2]])
dataset = dataset.reshape(1000,4)




plt.figure(figsize=(6,6))
plt.axvline(0, color='black')
plt.axhline(0, color='black')
plt.plot(x,y, 'bo')
plt.xlabel('X position')
plt.ylabel('Y Position')
plt.title('Dataset of endpoints')
plt.savefig('dataset/xy1000.png')


# scaler = MinMaxScaler()
dataset_pd = pd.DataFrame(dataset, columns=['x','y','theta1','theta2'])
# df_norm = pd.DataFrame(scaler.fit_transform(dataset_pd), columns=dataset_pd.columns)
df_norm = dataset_pd

split = int(dataset_pd.shape[0]*0.8)

train_df = df_norm.iloc[:split, :]
test_df = df_norm.iloc[split:, :]



train_df.to_json('dataset/dataset1000_train.json', orient='records', indent=4)
test_df.to_json('dataset/dataset1000_test.json', orient='records', indent=4)


