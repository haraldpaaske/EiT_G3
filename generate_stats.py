import pickle
import numpy as np
import time
import matplotlib.pyplot as plt


def random_point_in_sphere(radius=2.826):
    """
    Generate a random point uniformly distributed within a sphere of the given radius.
    """
    # Generate a random direction by normalizing a 3D Gaussian vector
    direction = np.random.normal(0, 1, 3)
    direction /= np.linalg.norm(direction)
    # Generate a random radius with uniform distribution in volume
    r = radius * (np.random.random() ** (1 / 3))
    return r * direction


def random_query_point(perturb_radius=0.1, positions=None):
    """
    Generate a query point by selecting a random reachable candidate from the database
    and adding a small random perturbation. This ensures the query is near a reachable point.

    Parameters:
        perturb_radius (float): Maximum perturbation radius.
        positions (np.array): Array of candidate positions.

    Returns:
        np.array: A query point in 3D.
    """
    # Choose a random candidate from the database
    idx = np.random.randint(len(positions))
    base_point = positions[idx]
    # Add a small random offset
    offset = random_point_in_sphere(perturb_radius)
    return base_point + offset


# Load stored NumPy arrays (reachable candidate positions)
positions = np.load("coordinates.npy")
angles = np.load("angles.npy")

# Load KDTree
with open("kdtree.pkl", "rb") as f:
    kdtree = pickle.load(f)

num_iterations = 1000
times = []  # to store execution times in microseconds
dists = []  # to store Euclidean distances

# Run 1000 queries using random points near the candidate positions.
# Adjust the perturbation radius to control how far from a candidate the queries can be.
for _ in range(num_iterations):
    query_point = random_query_point(perturb_radius=0.1, positions=positions)

    # Start timer
    start_time = time.perf_counter()
    # Perform nearest neighbor search using the KDTree
    dist, index = kdtree.query(query_point)
    # Stop timer
    end_time = time.perf_counter()

    # Calculate elapsed time in microseconds and record it
    elapsed_time = (end_time - start_time) * 1e6
    times.append(elapsed_time)
    dists.append(dist)

# Convert lists to NumPy arrays for further processing
times = np.array(times)
dists = np.array(dists)

# Convert distances from meters to centimeters
dists_cm = dists * 100

# Calculate average and median values in centimeters
avg_time = times.mean()
avg_dist_cm = dists_cm.mean()
median_time = np.median(times)
median_dist_cm = np.median(dists_cm)
print(f"Average Execution Time: {avg_time:.4f} µs")
print(f"Average Euclidean Distance: {avg_dist_cm:.4f} cm")
print(f"Median Euclidean Distance: {median_dist_cm:.4f} cm")

# Plotting the results with distances in centimeters:
plt.figure(figsize=(8, 6))
plt.scatter(dists_cm, times, alpha=0.5, label='Individual Queries')
plt.scatter(median_dist_cm, median_time, color='red', s=100, label='Median (Speed & Distance)')
plt.xlabel('Euclidean Distance (cm)')
plt.ylabel('Query Execution Time (µs)')
plt.title('Nearest Neighbor Query: Execution Time vs Euclidean Distance (1000 Iterations)')
plt.ylim(8, 15)
plt.legend()
plt.grid(True)
plt.show()