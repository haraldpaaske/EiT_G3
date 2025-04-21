import pickle
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# Force interactive mode
pio.renderers.default = "browser"

def rotate_x(vec, angle):
    """Rotate a vector around the X-axis by 'angle' radians."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.dot(R, vec)


def rotate_y(vec, angle):
    """Rotate a vector around the Y-axis by 'angle' radians."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.dot(R, vec)

def rotate_z(vec, angle):
    """Rotate a vector around the Z-axis by 'angle' radians."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return np.dot(R, vec)


def forward_kinematics(joint_angles, link_lengths=[1.045, 1.3, 1.025, 0.29]):
    """Computes all joint positions using forward kinematics."""
    A1, A2, A3, A4, A5 = joint_angles  # Already in radians
    L1, L2, L3, L4 = link_lengths  # Link lengths

    # Base rotation (Yaw around Z-axis)
    c1, s1 = np.cos(A1), np.sin(A1)

    # Shoulder and Elbow (Pitch around Y-axis)
    c2, s2 = np.cos(A2), np.sin(A2)
    c23, s23 = np.cos(A2 + A3), np.sin(A2 + A3)

    # Define base with slope: Shift shoulder joint 0.5 units
    shoulder_offset = np.array([0.5 * np.cos(A1), 0.5 * np.sin(A1), L1])

    joint1 = np.array([0, 0, 0])
    joint2 = joint1 + shoulder_offset
    joint3 = joint2 + np.array([L2 * c1 * c2, L2 * s1 * c2, L2 * s2])  # Elbow
    joint4 = joint3 + np.array([L3 * c1 * c23, L3 * s1 * c23, L3 * s23])  # Wrist Base

    #Apply full transformations to Joint 5 (inherit previous rotations)
    wrist_offset = np.array([L4, 0, 0])  # Initial wrist offset
    rotated_wrist = rotate_z(wrist_offset, A5)  # Apply wrist roll (A4)
    rotated_wrist = rotate_x(rotated_wrist, A4)  # Apply wrist tilt (A5)

    #Apply base rotation (A1) so joint5 fully inherits it
    rotated_wrist = rotate_z(rotated_wrist, A1)

    joint5 = joint4 + rotated_wrist  #Now properly follows full kinematic chain

    return [joint1, joint2, joint3, joint4, joint5]  #All joints are now computed sequentially

def plot_arm(joint_positions_nearest, target_position):
    """Plots the stored nearest arm and the target point."""
    fig = go.Figure()

    x_nearest, y_nearest, z_nearest = zip(*joint_positions_nearest)
    fig.add_trace(go.Scatter3d(x=x_nearest, y=y_nearest, z=z_nearest, mode='lines+markers',
                               marker=dict(size=5, color='red'), line=dict(width=5, color='red'),
                               name="Nearest Stored"))

    fig.add_trace(go.Scatter3d(
        x=[target_position[0]], y=[target_position[1]], z=[target_position[2]],
        mode='markers', marker=dict(size=8, color='green'), name="Target"
    ))

    fig.update_layout(
        title="Robotic Arm: Nearest Stored vs Target",
        scene=dict(xaxis=dict(range=[-3, 3]), yaxis=dict(range=[-3, 3]), zaxis=dict(range=[-3, 3]))
    )
    fig.show()

positions = np.load("coordinates.npy")
angles = np.load("angles.npy")
with open("kdtree.pkl", "rb") as f:
    kdtree = pickle.load(f)

def nearest_neighbor(query_point):
    dist, index = kdtree.query(query_point)
    nearest_point = positions[index]
    nearest_angles = angles[index]
    return nearest_point, nearest_angles

if __name__ == "__main__":
    query_point = [-1.6438, -1.0806,  1.7167]
    nearest_point, nearest_angles = nearest_neighbor(query_point)

    # Ensure angles are used correctly (convert if needed)  # Only if stored in radians
    joint_positions_nearest = forward_kinematics(nearest_angles)  # Corrected angle input
    plot_arm(joint_positions_nearest, query_point)

    print("Target Point", query_point)
    print("Nearest Point from KDTree:", nearest_point)
    print("Nearest Angles from KDTree:", nearest_angles)