import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# Set default Plotly renderer
pio.renderers.default = "browser"

# Arm segment lengths
L1 = 1.0  # Base to shoulder
L2 = 1.0  # Shoulder to elbow
L3 = 1.0  # Elbow to wrist
L4 = 0.5  # Wrist to hand (end-effector)

# Prime number for modular arithmetic (Not used in this implementation)
Prime = 1223


def inverse_kinematics(x, y, z, L1, L2, L3, L4):
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

    # Adjusted distance (hand should reach the target, not the elbow)
    distance_to_hand = np.sqrt(horizontal**2 + (z - L1)**2) - L4  # Adjust for wrist length

    if distance_to_hand > (L2 + L3) or distance_to_hand < abs(L2 - L3):
        print(f"Target ({x}, {y}, {z}) is out of reach.")
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
    theta5 = 0  # This can be adjusted as needed

    # Convert angles to degrees
    theta1_deg = np.degrees(theta1)
    theta2_deg = np.degrees(theta2)
    theta3_deg = np.degrees(theta3)
    theta4_deg = np.degrees(theta4)
    theta5_deg = np.degrees(theta5)

    # Save data to a file
    with open("../robot_arm_database.txt", "a") as file:
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


def is_target_reachable(x, y, z, L1, L2, L3, L4):
    """ Checks if a target (x, y, z) is within the reachable workspace. """
    d_horizontal = np.sqrt(x**2 + y**2)
    d_target = np.sqrt(d_horizontal**2 + (z - L1)**2)  # Compute actual distance to target

    R_min = abs(L2 - (L3 + L4))  # Fully folded reach
    R_max = L2 + L3 + L4         # Fully extended reach

    return R_min <= d_target <= R_max


def generate_random_point(L1, L2, L3, L4, num_samples=100):
    """ Generates a random reachable (x, y, z) target point. """
    while True:
        x, y, z = np.random.uniform(-L2 - L3, L2 + L3, 3)  # Generate points in a reasonable range

        if is_target_reachable(x, y, z, L1, L2, L3, L4):
            return round(x,2), round(y,2), round(z,2)



if __name__ == "__main__":
    num_samples = 10000  # Number of points to generate
    with open("../robot_arm_database.txt", "w") as file:
        file.write("Target (x, y, z) -> Joint Angles (theta1, theta2, theta3, theta4, theta5)\n")

        for _ in range(num_samples):
            x, y, z = generate_random_point(L1, L2, L3, L4)

            result = inverse_kinematics(x, y, z, L1, L2, L3, L4)
            if result:
                angles = result["Joint Angles"]
                file.write(f"[{x}, {y}, {z}] -> [{angles[0]:.2f}, {angles[1]:.2f}, {angles[2]:.2f}, {angles[3]:.2f}, {angles[4]:.2f}]\n")

    print("Generated robot arm database with reachable target points and joint angles.")