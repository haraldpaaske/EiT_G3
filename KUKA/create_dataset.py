from forward_kin import forward_6dof
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import sympy as sm
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import MinMaxScaler
#Configurations of robotarm
α_i = [np.radians(90),
       0,
       np.radians(-90),
       np.radians(90),
       np.radians(-90),
       0]
d_i= [-50,
      -130,
      5.5,
      0,
      0,
      0,]
r_i = [104.5,
       0,
       0,
       102.5,
       0,
       23]

#Joint ranges
theta1_min, theta1_max =   np.radians(-185),  np.radians(185)  # Joint 1
theta2_min, theta2_max =   np.radians(-130),   np.radians(20)  # Joint 2
theta3_min, theta3_max =   np.radians(-100), np.radians(144)  # Joint 3
theta4_min, theta4_max =   np.radians(-350),  np.radians(350)  # Joint 4
theta5_min, theta5_max =   np.radians(-120),  np.radians(120)  # Joint 5
theta6_min, theta6_max =   np.radians(-350),  np.radians(350)  # Joint 6

#Number of points in dataset
num_samples = 30000

theta1 = np.random.uniform(theta1_min, theta1_max, num_samples)
theta2 = np.random.uniform(theta2_min, theta2_max, num_samples)
theta3 = np.random.uniform(theta3_min, theta3_max, num_samples)
theta4 = np.random.uniform(theta4_min, theta4_max, num_samples)
theta5 = np.random.uniform(theta5_min, theta5_max, num_samples)
theta6 = np.random.uniform(theta6_min, theta6_max, num_samples)



t1 = np.full((num_samples,), np.radians(180)) + theta1
t2 = np.full((num_samples,), np.radians(90)) + theta2
t3 = theta3
t4 = theta4
t5 = theta5
t6 = theta6

t = np.column_stack((t1, t2, t3, t4, t5, t6))
data_points = forward_6dof(t, α_i, r_i, d_i)



#Create dataset
#-------------------------------------------------------------------------------------------------

# x_vals, y_vals, z_vals, e1, e2, e3 = zip(*[(x, y, z, e1, e2, e3) for x, y, z, e1, e2, e3 in data_points])
x_vals, y_vals, z_vals, e1, e2, e3 = np.array([
    (x, y, z, e1, e2, e3) for x, y, z, e1, e2, e3 in data_points
]).T

dataset = np.array([x_vals, y_vals, z_vals, e1, e2, e3, 
                    theta1, theta2, theta3, theta4, theta5, theta6]).T  # Transpose to (num_samples, 12)

dataset_pd = pd.DataFrame(dataset, columns=['x','y','z','e1','e2', 'e3',
                                            'theta1','theta2','theta3','theta4','theta5','theta6'])
dataset_pd = dataset_pd.applymap(lambda x: float(x.evalf()) if isinstance(x, sm.Basic) else x)

# scaler = MinMaxScaler(feature_range=(-1,1))


train_set, val_set = train_test_split(dataset_pd, test_size=0.1, random_state=42)

# for col in dataset_pd.columns:
#     print(f"{col}: {dataset_pd[col].apply(lambda x: type(x)).unique()}")


os.makedirs(f'KUKA/data/dataset/dataset{num_samples}', exist_ok=True)
train_set.to_json(f'KUKA/data/dataset/dataset{num_samples}/train.json', orient='records', indent=4)
val_set.to_json(f'KUKA/data/dataset/dataset{num_samples}/val.json', orient='records', indent=4)




#Plot dataset for visualization
#-------------------------------------------------------------------------------------------------

# # Create a 3D plot
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')

# # Scatter plot
# ax.scatter(x_vals, y_vals, z_vals, c='b', marker='o', label="XYZ Points")

# # Labels and title
# ax.set_xlabel(r"$X^{B}$")
# ax.set_ylabel(r"$Y^{B}$")
# ax.set_zlabel(r"$Z^{B}$")
# # ax.set_title("3D Plot of XYZ Coordinates")

# # Show legend
# ax.legend()

# # Show plot
# plt.savefig('KUKA/data/datapoints_100_new.png')
# # plt.show()





