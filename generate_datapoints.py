import numpy as np
from scipy.stats import qmc


def generate_random_joint_angles():
    """Generates random joint angles with proper motion constraints."""
    A1 = np.random.uniform(-3.2289, 3.2289)  # Base spins (Yaw)
    A2 = np.random.uniform(-0.3490, 2.2689)  # Shoulder moves up/down (Pitch)
    A3 = np.random.uniform(-2.5133, 1.7453)  # Elbow moves up/down (Pitch)
    A4 = np.random.uniform(-6.1087, 6.1087)  # Wrist spins around itself (Roll)
    A5 = np.random.uniform(-2.0944, 2.0944)  # Wrist moves up/down (Pitch, depends on A4)
    A6 = np.random.uniform(-6.1087, 6.1087)  # End rotates (Yaw or Pitch, depends on A4 & A5)
    return np.around([A1, A2, A3, A4, A5], decimals=4)


def rotate_x(vec, angle):
    """Rotate a vector around the X-axis by 'angle' radians."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[1, 0, 0],
                  [0, c, -s],
                  [0, s, c]])
    return np.dot(R, vec)


def rotate_y(vec, angle):
    """Rotate a vector around the Y-axis by 'angle' radians."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, 0, s],
                  [0, 1, 0],
                  [-s, 0, c]])
    return np.dot(R, vec)


def rotate_z(vec, angle):
    """Rotate a vector around the Z-axis by 'angle' radians."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0],
                  [s, c, 0],
                  [0, 0, 1]])
    return np.dot(R, vec)


def forward_kinematics(joint_angles, link_lengths=[1.045, 1.3, 1.025, 0.29]):
    """
    Computes the 3D position of the end-effector using forward kinematics.
    """
    A1, A2, A3, A4, A5 = joint_angles  # Angles in radians
    L1, L2, L3, L4 = link_lengths  # Link lengths

    # Base rotation (Yaw around Z-axis)
    c1, s1 = np.cos(A1), np.sin(A1)

    # Shoulder and Elbow (Pitch around Y-axis)
    c2, s2 = np.cos(A2), np.sin(A2)
    c23, s23 = np.cos(A2 + A3), np.sin(A2 + A3)

    # Define base with offset: shift shoulder joint 0.5 units
    shoulder_offset = np.array([0.5 * np.cos(A1), 0.5 * np.sin(A1), L1])

    joint1 = np.array([0, 0, 0])
    joint2 = joint1 + shoulder_offset
    joint3 = joint2 + np.array([L2 * c1 * c2, L2 * s1 * c2, L2 * s2])
    joint4 = joint3 + np.array([L3 * c1 * c23, L3 * s1 * c23, L3 * s23])

    # Apply wrist rotations:
    wrist_offset = np.array([L4, 0, 0])  # Initial wrist offset
    rotated_wrist = rotate_z(wrist_offset, A5)  # Wrist roll (using A5)
    rotated_wrist = rotate_x(rotated_wrist, A4)  # Wrist tilt (using A4)
    rotated_wrist = rotate_z(rotated_wrist, A1)  # Base rotation (inherit overall orientation)

    joint5 = joint4 + rotated_wrist
    return joint5


def farthest_point_sampling(points, num_points):
    """
    Selects a subset of points using a greedy farthest point sampling (FPS) algorithm.

    Parameters:
        points (np.array): An array of shape (N, 3) containing the candidate points.
        num_points (int): The number of points to select.

    Returns:
        np.array: Indices of the selected points.
    """
    N = points.shape[0]
    selected_indices = []
    # Start with the first candidate (or choose a random index if preferred)
    selected_indices.append(0)
    # Initialize distances to a large value for all candidates
    distances = np.full(N, np.inf)

    for i in range(1, num_points):
        # Get the last selected point
        last_selected = points[selected_indices[-1]]
        # Compute Euclidean distances from the last selected point to all candidates
        dist = np.linalg.norm(points - last_selected, axis=1)
        # Update the minimum distance for each candidate
        distances = np.minimum(distances, dist)
        # Select the candidate with the maximum distance from the current set
        next_index = np.argmax(distances)
        selected_indices.append(next_index)
        if i % 1000 == 0:
            print(f"FPS progress: Selected {i} / {num_points} points")
    return np.array(selected_indices)


if __name__ == "__main__":
    # Link lengths for the robotic arm
    link_lengths = [1.045, 1.3, 1.025, 0.29]

    # Define the joint limits for the 5 joints
    bounds = np.array([
        [-3.2289, 3.2289],  # A1
        [-0.3490, 2.2689],  # A2
        [-2.5133, 1.7453],  # A3
        [-6.1087, 6.1087],  # A4
        [-2.0944, 2.0944]  # A5
    ])

    # Parameters for candidate and optimized set sizes
    n_candidates = 1000000  # Oversampled candidate set
    n_optimized = 500000  # Final number of points to select

    # Create a Sobol sampler for 5 dimensions (joint space)
    sampler = qmc.Sobol(d=5, scramble=False)
    m = int(np.ceil(np.log2(n_candidates)))  # Use a power-of-2 number of samples
    samples = sampler.random_base2(m=m)
    samples = samples[:n_candidates]  # Trim to exactly n_candidates

    # Scale the Sobol samples to the joint angle ranges
    candidate_joint_angles = qmc.scale(samples, bounds[:, 0], bounds[:, 1])

    # Compute corresponding end-effector positions
    candidate_positions = np.empty((n_candidates, 3))
    for i in range(n_candidates):
        candidate_positions[i] = forward_kinematics(candidate_joint_angles[i], link_lengths)
    print("Candidate positions computed.")

    # --- Outlier Filtering ---
    center = np.mean(candidate_positions, axis=0)
    dists_center = np.linalg.norm(candidate_positions - center, axis=1)
    threshold = np.percentile(dists_center, 95)  # Keep points within 95th percentile
    inlier_mask = dists_center <= threshold
    filtered_positions = candidate_positions[inlier_mask]
    filtered_joint_angles = candidate_joint_angles[inlier_mask]
    print(f"Filtered candidates: {filtered_positions.shape[0]} out of {n_candidates}")

    # Choose which candidate set to use for FPS
    if filtered_positions.shape[0] < n_optimized:
        print("Not enough filtered candidates; using full candidate set for FPS.")
        fps_positions = candidate_positions
        fps_joint_angles = candidate_joint_angles
    else:
        fps_positions = filtered_positions
        fps_joint_angles = filtered_joint_angles

    # --- Farthest Point Sampling ---
    print("Starting farthest point sampling (FPS)...")
    selected_indices = farthest_point_sampling(fps_positions, n_optimized)
    print("FPS complete.")

    optimized_positions = fps_positions[selected_indices]
    optimized_joint_angles = fps_joint_angles[selected_indices]

    # --- Write to file ---
    output_file = "datasets/FPS500_datapoints.txt"
    with open(output_file, "w") as file:
        for i, (angles, position) in enumerate(zip(optimized_joint_angles, optimized_positions)):
            angles_str = np.array2string(angles, precision=4, separator=', ')
            position_str = np.array2string(position, precision=4, separator=', ')
            file.write(f"{position_str} : {angles_str}\n")
            if i % 1000 == 0:
                print(f"File writing progress: Processed {i} / {n_optimized} points")
    print(f"Done writing optimized data to {output_file}")