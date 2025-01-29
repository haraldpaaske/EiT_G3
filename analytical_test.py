import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def inverse_kinematics(x, y, z, L1=1.0, L2=1.0, L3=1.0):
    """
    Computes the inverse kinematics for a 3-DOF robotic arm in 3D space.

    Args:
        x, y, z: Target coordinates.
        L1, L2, L3: Lengths of the arm segments.

    Returns:
        A dictionary containing the possible joint angles (if a solution exists),
        along with the computed joint positions.
    """

    # Compute base rotation theta1
    theta1 = np.arctan2(y, x)  # Rotation around the Z-axis

    # Project target onto the XY plane
    r = np.sqrt(x**2 + y**2)  # Horizontal distance from base to target
    d = np.sqrt(r**2 + (z - L1)**2)  # Distance from shoulder to target

    # Check if the target is within reachable limits
    if d > (L2 + L3) or d < abs(L2 - L3):
        print("Target is out of reach!")
        return None

    # Law of Cosines for theta3 (elbow angle)
    cos_theta3 = (L2**2 + L3**2 - d**2) / (2 * L2 * L3)
    theta3 = np.pi - np.arccos(np.clip(cos_theta3, -1.0, 1.0))  # Corrected to ensure proper bending

    # Law of Cosines for theta2 (shoulder angle)
    cos_theta2 = (L2**2 + d**2 - L3**2) / (2 * L2 * d)
    theta2 = np.arctan2(z - L1, r) - np.arccos(np.clip(cos_theta2, -1.0, 1.0))  # Corrected formula

    # Convert radians to degrees for readability
    theta1_deg = np.degrees(theta1)
    theta2_deg = np.degrees(theta2)
    theta3_deg = np.degrees(theta3)

    # Compute joint positions for visualization
    joint1 = np.array([0, 0, L1])  # Base position
    joint2 = joint1 + np.array([
        L2 * np.cos(theta1) * np.cos(theta2),
        L2 * np.sin(theta1) * np.cos(theta2),
        L2 * np.sin(theta2)
    ])

    joint3 = joint2 + np.array([
        L3 * np.cos(theta1) * np.cos(theta2 + theta3),
        L3 * np.sin(theta1) * np.cos(theta2 + theta3),
        L3 * np.sin(theta2 + theta3)
    ])

    return {
        "Theta1 (Base Rotation)": theta1_deg,
        "Theta2 (Shoulder)": theta2_deg,
        "Theta3 (Elbow)": theta3_deg,
        "Joint Positions": [joint1, joint2, joint3]
    }

def plot_robot_arm(joint_positions, target_position, L1, L2, L3):
    """
    Plots the robotic arm in 3D space.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract joint coordinates
    x_coords = [0, joint_positions[0][0], joint_positions[1][0], joint_positions[2][0]]
    y_coords = [0, joint_positions[0][1], joint_positions[1][1], joint_positions[2][1]]
    z_coords = [0, joint_positions[0][2], joint_positions[1][2], joint_positions[2][2]]

    ax.plot(x_coords, y_coords, z_coords, '-o', markersize=8, label="Robot Arm", color='blue')
    ax.scatter(x_coords, y_coords, z_coords, color='red', s=80, label="Joints")
    ax.scatter(target_position[0], target_position[1], target_position[2], color='green', s=100, label="Target")

    limit = sum([L1, L2, L3]) + 0.5
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([0, limit])

    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("3D Robot Arm Inverse Kinematics")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    x, y, z = map(float, input("Enter target coordinates (x y z): ").split())
    result = inverse_kinematics(x, y, z)
    if result:
        print(result)
        plot_robot_arm(result["Joint Positions"], (x, y, z), L1=1.0, L2=1.0, L3=1.0)