import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# Force interactive mode
pio.renderers.default = "browser"

def inverse_kinematics(x, y, z, L1=1.0, L2=1.0, L3=1.0, L4=1.0):
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

    # Projection on the XY plane
    horizontal = np.sqrt(x**2 + y**2)

    # Adjusted distance: The hand should reach the target, not the elbow
    distance_to_hand = np.sqrt(horizontal**2 + (z - L1)**2) - L4  # Adjust for wrist length

    if distance_to_hand > (L2 + L3) or distance_to_hand < abs(L2 - L3):
        print("Target is out of reach.")
        return None

    # Compute theta3 (Elbow angle) using the Law of Cosines
    cos_theta3 = (L2**2 + L3**2 - distance_to_hand**2) / (2 * L2 * L3)
    theta3 = np.arccos(np.clip(cos_theta3, -1.0, 1.0))

    # Compute theta2 (Shoulder angle)
    angle_to_target = np.arctan2(z - L1, horizontal)
    cos_theta2 = (L2**2 + distance_to_hand**2 - L3**2) / (2 * L2 * distance_to_hand)
    theta2 = angle_to_target - np.arccos(np.clip(cos_theta2, -1.0, 1.0))

    # Wrist rotation (ensuring alignment with target)
    theta4 = -theta3  # Wrist should compensate for elbow rotation to keep hand alignment

    # Hand vertical movement (fine-tuning)
    theta5 = 0

    # Convert angles to degrees
    theta1_deg = np.degrees(theta1)
    theta2_deg = np.degrees(theta2)
    theta3_deg = np.degrees(theta3)
    theta4_deg = np.degrees(theta4)
    theta5_deg = np.degrees(theta5)

    # Save data to a file
    with open("robot_arm_database.txt", "a") as file:
        file.write(f"[{x}, {y}, {z}], [{theta1_deg:.2f}, {theta2_deg:.2f}, {theta3_deg:.2f}, {theta4_deg:.2f}, {theta5_deg:.2f}]\n")

    # Compute joint positions (adjusted for base being at the lowest point)
    joint1 = np.array([0, 0, 0])  # Base joint (now at the lowest point)

    # Shoulder position (previously the base location)
    joint2 = joint1 + np.array([0, 0, L1])  # Moves up to the original base location

    # Compute Elbow Position
    joint3 = joint2 + np.array([
        L2 * np.cos(theta1) * np.cos(theta2),
        L2 * np.sin(theta1) * np.cos(theta2),
        L2 * np.sin(theta2)
    ])

    # Compute Wrist Position (ensuring the hand reaches the target)
    joint4 = joint3 + np.array([
        L3 * np.cos(theta1) * np.cos(theta2 + theta3),
        L3 * np.sin(theta1) * np.cos(theta2 + theta3),
        L3 * np.sin(theta2 + theta3)
    ])

    # Hand position (final end-effector reaching the target)
    joint5 = np.array([x, y, z])  # Hand reaches the target

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
        scene=dict(xaxis=dict(range=[-2, 2]), yaxis=dict(range=[-2, 2]), zaxis=dict(range=[-2, 2]))
    )

    fig.show()


def generate_random_point(max_sum=4):
    while True:
        x, y, z = np.random.uniform(0, max_sum, 3)
        if (x + y + z) < max_sum:
            return round(x,2), round(y,2), round(z,2)



if __name__ == "__main__":
    for i in range(5):  # Change this to 100 for large-scale testing
        x, y, z = map(float, input("Enter target coordinates (x, y, z): ").split(","))

        result = inverse_kinematics(x, y, z)
        if result:
            plot_robot_arm(result["Joint Positions"], (x, y, z))