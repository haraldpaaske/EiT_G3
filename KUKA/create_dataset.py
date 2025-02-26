from forward_kin import forward_6dof
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import sympy as sm
#Configurations of robotarm
α_i = [0,90,90,0,-90,-90,90,-90,0]
d_i= [0,0.479,0.5,0.178,0,0.0557,0.536,0,0.237]
r_i = [0.566,-0.067,0,1.3,0.489,0,0,0,0]

#Joint ranges
theta1_min, theta1_max =   -185,  185  # Joint 1
theta2_min, theta2_max =   -130,   20  # Joint 2
theta3_min, theta3_max =   -100, 144  # Joint 3
theta4_min, theta4_max =   -350,  350  # Joint 4
theta5_min, theta5_max =   120,  120  # Joint 5
theta6_min, theta6_max =   -350,  350  # Joint 6

#Number of points in dataset
num_samples = 1000

theta1 = np.random.uniform(theta1_min, theta1_max, num_samples)
theta2 = np.random.uniform(theta2_min, theta2_max, num_samples)
theta3 = np.random.uniform(theta3_min, theta3_max, num_samples)
theta4 = np.random.uniform(theta4_min, theta4_max, num_samples)
theta5 = np.random.uniform(theta5_min, theta5_max, num_samples)
theta6 = np.random.uniform(theta6_min, theta6_max, num_samples)



t1 = np.full((num_samples,), 0)
t2 = np.full((num_samples,), 90) + theta1
t3 = np.full((num_samples,), 90)
t4 = theta2
t5 = np.full((num_samples,), 90) + theta3
t6 = np.full((num_samples,), -90)
t7 = np.full((num_samples,), 90) + theta4
t8= np.random.uniform(theta5_min, theta5_max, num_samples)
t9 = np.random.uniform(theta6_min, theta6_max, num_samples)

t = np.column_stack((t1, t2, t3, t4, t5, t6, t7, t8, t9))
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

# for col in dataset_pd.columns:
#     print(f"{col}: {dataset_pd[col].apply(lambda x: type(x)).unique()}")

dataset_pd.to_json('KUKA/data/dataset/dataset1000.json', orient='records', indent=4)


#Plot dataset for visualization
#-------------------------------------------------------------------------------------------------

# Create a 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(x_vals, y_vals, z_vals, c='b', marker='o', label="XYZ Points")

# Labels and title
ax.set_xlabel(r"$X^{B}$")
ax.set_ylabel(r"$Y^{B}$")
ax.set_zlabel(r"$Z^{B}$")
# ax.set_title("3D Plot of XYZ Coordinates")

# Show legend
ax.legend()

# Show plot
plt.savefig('KUKA/data/datapoints_1000_new.png')
# plt.show()





