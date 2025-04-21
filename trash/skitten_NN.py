import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Function for forward kinematics (3-DOF planar arm example)
def forward_kinematics(q, link_lengths):
    x = link_lengths[0] * np.cos(q[0]) + link_lengths[1] * np.cos(q[0] + q[1]) + link_lengths[2] * np.cos(
        q[0] + q[1] + q[2])
    y = link_lengths[0] * np.sin(q[0]) + link_lengths[1] * np.sin(q[0] + q[1]) + link_lengths[2] * np.sin(
        q[0] + q[1] + q[2])
    z = 0  # This example assumes planar motion; z is constant
    return np.array([x, y, z])


# Generate data for training
def generate_data(num_samples, joint_limits, link_lengths):
    data = []
    for _ in range(num_samples):
        # Random joint angles within limits
        q = [np.random.uniform(*joint_limits[i]) for i in range(len(joint_limits))]
        # Forward kinematics to compute end-effector position
        position = forward_kinematics(q, link_lengths)
        data.append((position, q))
    return data


# Define the neural network
class IKNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(IKNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)


def test_model(target_position, link_lengths):
    # Load the saved model
    model = IKNet(input_dim=3, hidden_dim=128, output_dim=3)
    model.load_state_dict(torch.load("ik_model.pth"))
    model.eval()

    # Convert target position to tensor
    target_tensor = torch.tensor(target_position, dtype=torch.float32).unsqueeze(0)

    # Predict joint angles
    with torch.no_grad():
        predicted_angles = model(target_tensor).squeeze(0).numpy()

    # Compute the forward kinematics for validation
    end_effector_position = forward_kinematics(predicted_angles, link_lengths)

    return predicted_angles, end_effector_position


def plot_robot_arm_3d(joint_angles, link_lengths, target_position=None, end_effector_position=None):
    """
    Visualizes the robotic arm in 3D space.

    Parameters:
    - joint_angles: List of joint angles (q1, q2, q3) in radians.
    - link_lengths: List of link lengths [L1, L2, L3].
    - target_position: Optional, the target position [x, y, z].
    - end_effector_position: Optional, the calculated end-effector position [x, y, z].
    """
    # Compute joint positions
    x1 = link_lengths[0] * np.cos(joint_angles[0])
    y1 = link_lengths[0] * np.sin(joint_angles[0])
    z1 = 0  # Assume planar motion for simplicity

    x2 = x1 + link_lengths[1] * np.cos(joint_angles[0] + joint_angles[1])
    y2 = y1 + link_lengths[1] * np.sin(joint_angles[0] + joint_angles[1])
    z2 = 0  # Still planar motion

    x3 = x2 + link_lengths[2] * np.cos(joint_angles[0] + joint_angles[1] + joint_angles[2])
    y3 = y2 + link_lengths[2] * np.sin(joint_angles[0] + joint_angles[1] + joint_angles[2])
    z3 = 0  # Planar motion

    # Joint positions
    joints = np.array([[0, 0, 0], [x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])

    # Plot the robotic arm in 3D
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the arm
    ax.plot(joints[:, 0], joints[:, 1], joints[:, 2], '-o', label="Robot Arm", color='blue')
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='red', zorder=5, label="Joints")  # Mark joints

    # Plot the target position if provided
    if target_position is not None:
        ax.scatter(target_position[0], target_position[1], target_position[2], color='green', label="Target Position", s=100)

    # Plot the end-effector position if provided
    if end_effector_position is not None:
        ax.scatter(end_effector_position[0], end_effector_position[1], end_effector_position[2], color='orange', label="End Effector", s=100)

    # Set labels and limits
    ax.set_xlim([-sum(link_lengths) - 0.5, sum(link_lengths) + 0.5])
    ax.set_ylim([-sum(link_lengths) - 0.5, sum(link_lengths) + 0.5])
    ax.set_zlim([-1, 1])  # Z-axis limits (flat arm motion)
    ax.set_title("3D Visualization of Robotic Arm")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.legend()
    plt.show()


# Hyperparameters and setup
link_lengths = [1.0, 1.0, 1.0]  # Lengths of the links
joint_limits = [(-np.pi, np.pi), (-np.pi / 2, np.pi / 2), (-np.pi / 4, np.pi / 4)]  # Joint limits
num_samples = 50000
hidden_dim = 128
learning_rate = 0.001
num_epochs = 1000

# Generate training and testing data
data = generate_data(num_samples, joint_limits, link_lengths)
X = torch.tensor([d[0] for d in data], dtype=torch.float32)  # End-effector positions
y = torch.tensor([d[1] for d in data], dtype=torch.float32)  # Joint angles

# Split into training and testing datasets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Initialize the neural network
model = IKNet(input_dim=3, hidden_dim=hidden_dim, output_dim=3)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# Evaluate on test data
model.eval()
with torch.no_grad():
    
    test_loss = criterion(model(X_test), y_test)
    print(f"Test Loss: {test_loss.item():.6f}")

# Save the model and test data for further exploration
torch.save(model.state_dict(), "ik_model.pth")
np.savez("ik_test_data.npz", X_test=X_test.numpy(), y_test=y_test.numpy())

print("Model trained and test data saved for exploration.")

# Testing functionality
if __name__ == "__main__":
    # Example target positions
    target_positions = [
        [1.5, 0.5, 0],
        [0.5, 1.5, 0],
        [-1.0, -1.0, 0]
    ]

    print("\nTesting the model on example target positions:")
    for target_position in target_positions:
        predicted_angles, end_effector_position = test_model(target_position, link_lengths)
        print(f"\nTarget Position: {target_position}")
        print(f"Predicted Joint Angles: {predicted_angles}")
        print(f"End Effector Position (from predicted angles): {end_effector_position}")

        # Visualize the result in 3D
        plot_robot_arm_3d(predicted_angles, link_lengths, target_position, end_effector_position)