import numpy as np

# def scara_inverse_kinematics_3d(x, y, z, L1, L2, z_base=0):
#     """
#     Solves the inverse kinematics for a 3D SCARA robot.

#     Parameters:
#     x (float): Desired x position of the end-effector.
#     y (float): Desired y position of the end-effector.
#     z (float): Desired z position of the end-effector.
#     theta_end (float): Desired end-effector rotation (radians).
#     L1 (float): Length of the first arm segment.
#     L2 (float): Length of the second arm segment.
#     z_base (float): Fixed height of the robot base.

#     Returns:
#     tuple: (theta1, theta2, d, theta3) in radians
#     """

#     # Compute radial distance from the base
#     r = np.sqrt(x**2 + y**2)

#     # Check if the position is reachable
#     if r > (L1 + L2) or r < abs(L1 - L2):
#         raise ValueError("Target position is out of reach.")

#     # Solve for theta2 using the law of cosines
#     cos_theta2 = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
#     theta2 = np.arccos(np.clip(cos_theta2, -1.0, 1.0))  # Clip to avoid numerical errors

#     # Solve for theta1 using the law of tangents
#     k1 = L1 + L2 * np.cos(theta2)
#     k2 = L2 * np.sin(theta2)
#     theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)

#     # Solve for prismatic joint displacement (d)
#     d = z - z_base  # Difference from the base Z position

#     # Solve for theta3 (end-effector rotation)
#     # theta3 = theta_end - (theta1 + theta2)

#     return theta1, theta2, d

# # Example Usage:
# L1 = 1.0  # Length of link 1
# L2 = 0.8  # Length of link 2
# x_desired = 0.9757393069082316  # Target x position
# y_desired = -1.13315754905644  # Target y position
# z_desired = 0.0  # Target z position
# # theta_end_effector = np.radians(30)  # Target end-effector rotation in degrees

# theta1, theta2, d = scara_inverse_kinematics_3d(x_desired, y_desired, z_desired, L1, L2)

# print(f"θ1: {theta1}rad")
# print(f"θ2: {theta2}rad")
# print(f"d: {d:.2f} (linear Z movement)")
# # print(f"θ3: {np.degrees(theta3):.2f}°")



# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# def plot_scara_3d(theta1, theta2, d, L1, L2, z_base=0):
#     """
#     Plots a 3D SCARA robot based on joint angles and link lengths.

#     Parameters:
#     theta1 (float): First joint angle in radians.
#     theta2 (float): Second joint angle in radians.
#     d (float): Prismatic joint displacement (vertical movement).
#     theta3 (float): End-effector rotation in radians.
#     L1 (float): Length of the first arm segment.
#     L2 (float): Length of the second arm segment.
#     z_base (float): Fixed base height of the robot.
#     """

#     # Calculate joint positions
#     x0, y0, z0 = 0, 0, z_base  # Base position
#     x1 = L1 * np.cos(theta1)
#     y1 = L1 * np.sin(theta1)
#     z1 = z_base  # First joint height remains the same

#     x2 = x1 + L2 * np.cos(theta1 + theta2)
#     y2 = y1 + L2 * np.sin(theta1 + theta2)
#     z2 = z_base  # Second joint height remains the same

#     x3, y3, z3 = x2, y2, z2 + d  # End-effector position

#     # Plotting the SCARA robot
#     fig = plt.figure(figsize=(8, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     # Plot links
#     ax.plot([x0, x1], [y0, y1], [z0, z1], 'ro-', label="Link 1")  # Base to Joint 1
#     ax.plot([x1, x2], [y1, y2], [z1, z2], 'bo-', label="Link 2")  # Joint 1 to Joint 2
#     ax.plot([x2, x3], [y2, y3], [z2, z3], 'go-', label="Prismatic Joint")  # Joint 2 to End-effector

#     # Plot joints
#     ax.scatter([x0, x1, x2, x3], [y0, y1, y2, y3], [z0, z1, z2, z3], c='k', marker='o', s=100)

#     # Labels
#     ax.set_xlabel("X-axis")
#     ax.set_ylabel("Y-axis")
#     ax.set_zlabel("Z-axis")
#     ax.set_title("SCARA Robot Visualization")

#     # Limits
#     ax.set_xlim([-L1 - L2, L1 + L2])
#     ax.set_ylim([-L1 - L2, L1 + L2])
#     ax.set_zlim([z_base - 0.5, z_base + d + 0.5])

#     ax.legend()
#     plt.show()






# # Visualize SCARA arm
# plot_scara_3d(theta1, theta2, d, L1, L2)




def IK(x, y, z):
    L1 = 1.0
    L2 = 0.8

    theta_2 = np.arccos((x**2+y**2-L1**2-L2**2)/(2*L1*L2))
    theta_1 = np.arctan(y/x)-np.arctan(L2*np.sin(theta_2)/(L1+L2*np.cos(theta_2)))
    d = z

    print(theta_1, theta_2, d)


    #forward
    x1 = L1*np.cos(theta_1)+L2*np.cos(theta_1+theta_2)
    x2 = L1*np.sin(theta_1)+L2*np.sin(theta_1+theta_2)
    
    print(x1, x2)
    
x = 1.6536141665826594
y= 0.6467277854267768
z = 0

IK(x,y,z)