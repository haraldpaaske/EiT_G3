import pickle
import numpy as np
import time  # Import time module for measuring execution time

# Load stored NumPy arrays
positions = np.load("coordinates.npy")
angles = np.load("angles.npy")

# Load KDTree
with open("kdtree.pkl", "rb") as f:
    kdtree = pickle.load(f)

# Example query
query_point = [1.194,  1.6441, 1.1567]  # Example 3D point

# Start the timer
start_time = time.perf_counter()
# Perform nearest neighbor search
dist, index = kdtree.query(query_point)
# Stop the timer
end_time = time.perf_counter()

# Compute execution time
elapsed_time = (end_time - start_time) * 1e6  # Convert to nanoseconds

# Retrieve the nearest point and angles
nearest_point = positions[index]
nearest_angles = angles[index]

# Print results
print(f"Query Position: {query_point}")
print(f"Nearest Point: {nearest_point}")
print(f"Nearest Joint Angles: {nearest_angles}")
print(f"Distance: {dist*100}cm")
print(f"⏱️ Query Execution Time: {elapsed_time:.4f} microseconds")