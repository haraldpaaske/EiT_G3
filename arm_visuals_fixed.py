import time

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# Force interactive mode
pio.renderers.default = "browser"



def inverse_kinematics(x, y, z, L1=1.0, L2=1.0, L3=1.0, L4=0.5):
    """
    Computes inverse kinematics for a 5-DOF robotic arm:
    - Rotating base
    - Shoulder joint (vertical movement)
    - Elbow joint (vertical movement)
    - Wrist rotation
    - Hand vertical movement
    """

    # Base rotation (Yaw)
    theta1 = np.arctan2(y, x)

    xwrist = x - L4 * np.cos(theta1)
    ywrist = y - L4 * np.sin(theta1)
    zwrist = z 
    # Projection on the XY plane
    horizontal = np.sqrt(xwrist**2 + ywrist**2)

    distance_to_wrist = np.sqrt(xwrist**2 + ywrist**2 + (zwrist-L1)**2)
    # Adjusted distance: The hand should reach the target, not the elbow
    #distance_to_hand = np.sqrt(horizontal**2 + (z - L1)**2)   # Adjust for wrist length

    if distance_to_wrist > (L2 + L3) or distance_to_wrist < abs(L2 - L3):
        print("Target is out of reach.")
        return None


    # Compute angles
    theta2 = np.arctan2(zwrist - L1, horizontal) - np.arccos((L2**2 + distance_to_wrist**2 - L3**2) / (2 * L2 * distance_to_wrist))
    theta3 = np.pi - np.arccos((L2**2 + L3**2 - distance_to_wrist**2) / (2 * L2 * L3))
    theta4 = -(theta3+theta2)  # Wrist should compensate for elbow rotation to keep hand alignment
    theta5 = 0

    # Convert angles to degrees
    theta1_deg, theta2_deg, theta3_deg, theta4_deg, theta5_deg = map(np.degrees, [theta1, theta2, theta3, theta4, theta5])


    # Compute joint positions (adjusted for base being at the lowest point)
    joint1 = np.array([0, 0, 0])  # Base joint (now at the lowest point)
    joint2 = np.array([0, 0, L1])  # Moves up to the original base location

    joint3 = joint2 + np.array([
        L2 * np.cos(theta1) * np.cos(theta2),
        L2 * np.sin(theta1) * np.cos(theta2),
        L2 * np.sin(theta2)
    ])

    joint4 = joint3 + np.array([
        L3 * np.cos(theta1) * np.cos(theta2 + theta3),
        L3 * np.sin(theta1) * np.cos(theta2 + theta3),
        L3 * np.sin(theta2 + theta3)
    ])

    joint5 = joint4 + np.array([
    L4 * np.cos(theta1) * np.cos(theta2 + theta3 + theta4),
    L4 * np.sin(theta1) * np.cos(theta2 + theta3 + theta4),
    L4 * np.sin(theta2 + theta3 + theta4)
    ])

    print("L1: ", np.linalg.norm(joint2 - joint1))  # Should be L1 (1.0)
    print("L2: ", np.linalg.norm(joint3 - joint2))  # Should be L2 (1.0)
    print("L3: ", np.linalg.norm(joint4 - joint3))  # Should be L3 (1.0)
    print("L4: ", np.linalg.norm(joint5 - joint4))  # Should be L4 (0.5)

    return {
        "Joint Positions": [joint1, joint2, joint3, joint4, joint5],
        "Joint Angles": (theta1_deg, theta2_deg, theta3_deg, theta4_deg, theta5_deg)
    }


def plot_robot_arm(joint_positions, target_position):
    """
    Plots the robotic arm in 3D using Plotly.
    """
    x_coords, y_coords, z_coords = zip(*joint_positions)
    fig = go.Figure()

    # Plot arm movement
    fig.add_trace(
        go.Scatter3d(x=x_coords, y=y_coords, z=z_coords, mode='lines+markers',
                     marker=dict(size=5, color='blue'), line=dict(width=5, color='blue'), name="Robot Arm"))

    # Annotate joints
    labels = ["Base", "Shoulder", "Elbow", "Wrist", "Hand"]
    for i, (x, y, z) in enumerate(joint_positions):
        fig.add_trace(go.Scatter3d(x=[x], y=[y], z=[z], mode='text',
                                   text=[labels[i]], textposition="top center"))

    # Plot original target (green)
    fig.add_trace(go.Scatter3d(
        x=[target_position[0]], y=[target_position[1]], z=[target_position[2]],
        mode='markers', marker=dict(size=8, color='green'), name="Target"
    ))

    fig.update_layout(
        title="Robotic Arm Visualization",
        scene=dict(xaxis=dict(range=[-3,3]), yaxis=dict(range=[-3,3]), zaxis=dict(range=[-3,3]))
    )

    fig.show()




if __name__ == "__main__":
    #x, y, z = map(float, input("Enter target coordinates (x, y, z): ").split(","))
    start_time = time.perf_counter()
    x, y, z = [0.6098, 0.9133, 1.8862]
    result = inverse_kinematics(x,y,z)
    end_time = time.perf_counter()
    print((end_time - start_time) * 100000)
    if result:
        plot_robot_arm(result["Joint Positions"], (x, y, z))
        print(result["Joint Positions"])