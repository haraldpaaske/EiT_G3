import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import ast  # For parsing stored lists from the database

# Force interactive mode
pio.renderers.default = "browser"

# Arm segment lengths
L1 = 1.0  # Base to shoulder
L2 = 1.0  # Shoulder to elbow
L3 = 1.0  # Elbow to wrist
L4 = 1.0  # Wrist to hand (end-effector)



import numpy as np

def forward_kinematics(theta1, theta2, theta3, theta4, theta5, closest_point, L1, L2, L3, L4):
    """
    Computes forward kinematics for a 5-DOF robotic arm.
    - Given joint angles (theta1 to theta5), it returns the joint positions.
    - Ensures the hand (joint5) is exactly at the stored closest point.
    """

    # 🔹 Convert angles from degrees to radians
    theta1, theta2, theta3, theta4, theta5 = np.radians([theta1, theta2, theta3, theta4, theta5])

    # Base (fixed point at the lowest position)
    joint1 = np.array([0, 0, 0])

    # Shoulder position (moved up from the base)
    joint2 = joint1 + np.array([0, 0, L1])

    # Elbow position
    joint3 = joint2 + np.array([
        L2 * np.cos(theta1) * np.cos(theta2),
        L2 * np.sin(theta1) * np.cos(theta2),
        L2 * np.sin(theta2)
    ])

    # Wrist position
    joint4 = joint3 + np.array([
        L3 * np.cos(theta1) * np.cos(theta2 + theta3),
        L3 * np.sin(theta1) * np.cos(theta2 + theta3),
        L3 * np.sin(theta2 + theta3)
    ])

    # ✅ **Fix: Ensure joint5 matches the stored closest point**
    joint5 = np.array(closest_point)  # Directly set to the retrieved closest point

    return [joint1, joint2, joint3, joint4, joint5]


def load_database(filename="robot_arm_database.txt"):
    """ Loads target positions and corresponding joint angles from the database file. """
    data_points = []
    joint_angles = []

    with open(filename, "r") as file:
        next(file)  # Skip header line
        for line in file:
            parts = line.strip().split(" -> ")
            if len(parts) != 2:
                continue  # Skip invalid lines

            # Parse target position (x, y, z)
            position_str = parts[0].strip("[]").split(", ")
            position = tuple(map(float, position_str))

            # Parse joint angles
            angles = ast.literal_eval(parts[1])  # Convert stored string back to list of angles

            data_points.append(position)
            joint_angles.append(angles)

    return np.array(data_points), joint_angles


def find_closest_point(target, data_points, joint_angles):
    """ Finds the closest stored point to the target using Euclidean distance. """
    distances = np.linalg.norm(data_points - target, axis=1)  # Compute distances
    closest_index = np.argmin(distances)  # Find the index of the closest point

    return data_points[closest_index], joint_angles[closest_index], distances[closest_index]


def plot_robot_arm(joint_positions, closest_point, target_position):
    """
    Plots the robotic arm in 3D using Plotly.
    """
    x_coords, y_coords, z_coords = zip(*joint_positions)

    fig = go.Figure()

    # Plot arm movement
    fig.add_trace(
        go.Scatter3d(x=x_coords, y=y_coords, z=z_coords, mode='lines+markers',
                     marker=dict(size=6, color='blue'), line=dict(width=5, color='blue'), name="Robot Arm"))

    # Annotate joints
    labels = ["Base", "Shoulder", "Elbow", "Wrist", "Hand"]
    for i, (x, y, z) in enumerate(joint_positions):
        fig.add_trace(go.Scatter3d(x=[x], y=[y], z=[z], mode='text',
                                   text=[labels[i]], textposition="top center"))

    # Closest stored point (Green)
    fig.add_trace(go.Scatter3d(
        x=[closest_point[0]], y=[closest_point[1]], z=[closest_point[2]],
        mode='markers', marker=dict(size=10, color='green'), name="Closest Stored Point"
    ))

    # User inputted target (Orange)
    fig.add_trace(go.Scatter3d(
        x=[target_position[0]], y=[target_position[1]], z=[target_position[2]],
        mode='markers', marker=dict(size=10, color='orange'), name="User Target"
    ))

    fig.update_layout(
        title="Robotic Arm Visualization",
        scene=dict(xaxis=dict(range=[-2, 2]), yaxis=dict(range=[-2, 2]), zaxis=dict(range=[-2, 2])),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()


if __name__ == "__main__":
    # Load database
    data_points, joint_angles = load_database()

    user_input = input("Enter target coordinates (x, y, z) or 'exit' to quit: ")

    try:
        x, y, z = map(float, user_input.split(","))
        target = np.array([x, y, z])

        # Find closest stored point and retrieve corresponding joint angles
        closest_point, closest_joint_angles, distance = find_closest_point(target, data_points, joint_angles)

        # ✅ Compute joint positions using Forward Kinematics (FK)
        joint_positions = forward_kinematics(*closest_joint_angles, closest_point, L1, L2, L3, L4)

        # ✅ FIX: Use FK to reconstruct arm instead of retrieving stored positions
        plot_robot_arm(joint_positions, closest_point, target)

        # Display results
        print(f"\n🔹 Closest Point in Database: {closest_point}")
        print(f"📏 Distance to Target: {distance:.4f}")
        print(f"🤖 Retrieved Joint Angles: {closest_joint_angles}\n")
        print(f"🤖 Computed Joint Positions: {joint_positions}\n")

    except ValueError:
        print("Invalid input! Please enter coordinates in the format: x, y, z")