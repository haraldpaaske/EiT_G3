import numpy as np
import pickle
from scipy.spatial import KDTree
import re

# Load the txt file
txt_file = "datasets/random1M_datapoints.txt"  # Change this to your actual file path

# Lists to store extracted data
positions = []
angles = []

# Read and parse the file
with open(txt_file, "r") as file:
    for line in file:
        # Extract values using regex
        match = re.match(r"\[([\d\s.,-]+)\] : \[([\d\s.,-]+)\]", line)
        if match:
            position_str, angles_str = match.groups()

            # Convert string lists into actual Python lists of floats
            position = list(map(float, position_str.split(", ")))
            angle = list(map(float, angles_str.split(", ")))

            positions.append(position)
            angles.append(angle)

# Convert lists to NumPy arrays
positions = np.array(positions)
angles = np.array(angles)

# Save them as .npy files
np.save("coordinates.npy", positions)
np.save("angles.npy", angles)

print("✅ Coordinates and angles saved as NumPy arrays.")

# Build the KDTree using the coordinates
kdtree = KDTree(positions)

# Save KDTree to a file
with open("kdtree.pkl", "wb") as f:
    pickle.dump(kdtree, f)

print("✅ KDTree saved successfully!")
