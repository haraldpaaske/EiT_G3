import pickle
import numpy as np
import time  # Import time module for measuring execution time

# Load stored NumPy arrays
positions = np.load("coordinates.npy")
angles = np.load("angles.npy")

# Load KDTree
with open("kdtree.pkl", "rb") as f:
    kdtree = pickle.load(f)

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

# point to find
query_point = random_query_point(perturb_radius=0.1, positions=positions)
#query_point = [1.1,  1.2, 0.9]

# Warm-up call (ignore result)
_ = kdtree.query(random_query_point(0.1, positions))
_ = kdtree.query(random_query_point(0.1, positions))
_ = kdtree.query(random_query_point(0.1, positions))
_ = kdtree.query(random_query_point(0.1, positions))
_ = kdtree.query(random_query_point(0.1, positions))

# Actual timed query
start_time = time.perf_counter()
dist, index = kdtree.query(query_point)
end_time = time.perf_counter()

# Compute execution time
elapsed_time = (end_time - start_time) * 1e6  # Convert to microsec

# Retrieve the nearest point and angles
nearest_point = positions[index]
nearest_angles = angles[index]


"""
A hopefully flexible framework for doing inverse kinematics numerically using Denavit-Hartenberg convention.

The code requres the user to input parameters for a robotic arm, following the DH-procedure.
First, a schematic is drawn. from this scematic, a table is filled in. This table is in turn used to define
the transformations between coordinate systems between every joint.
The convention allows for these transforms to be multiplied by each other, accessing the arm tip transform relative to the global Csys.

The input state vector Q is related to the arm tip state vector X by the function f: Q -> X. The inverse kinematics is then calculated numerically
using the Jacobian-inverse, i.e. dQ = J_inv dX. The jacobian is calculated simply by the first order central difference method for first order derivative.

Using DLS for the inverse of the Jacobian will in theory allow for redundant joints, as well as protect the algorithm from situations where singularities occur.

"""


# function to insert table values into a single csys transform
def Csys_Transform_n(table_row):
    # inputs a Denavit Hartenberg table and converts it into a state matrix with homogeneous coordinates

    t, a, r, d = table_row

    return [[np.cos(t), -np.sin(t) * np.cos(a), np.sin(t) * np.sin(a), r * np.cos(t)],
            [np.sin(t), np.cos(t) * np.cos(a), -np.cos(t) * np.sin(a), r * np.sin(t)],
            [0, np.sin(a), np.cos(a), d],
            [0, 0, 0, 1]]


# function to append csys transforms into a list
def Combined_Csys_Transform(table, m=None):
    # input table of shape (# of joints, 4 (4 "types of motion"))
    # input the m-th joint to which the transformation goes to

    if m == None:
        m = len(table)

    combinedTransform = np.eye(4)

    for i in range(m):
        combinedTransform = combinedTransform @ Csys_Transform_n(table[i])

    return combinedTransform



# Make DH-table into a function f(Q) = X, Q is robot joint space, X is cartesian coordinates of arm tip

# f: Q -> X (translation)
def FK_position(Q, DH_table):

    pos = np.array(Combined_Csys_Transform(DH_table(Q)))[:3, 3]

    return pos.squeeze()

# f: Q -> X (orientation) # returns a vector with the end effector's orientation
def FK_orientation(Q, DH_table):

    ori_matrix = np.array(Combined_Csys_Transform(DH_table(Q)))[:-1, :-1].squeeze()

    ori = ori_matrix @ np.array([0, 0, 1])

    return ori

# f: Q -> X (combined) # Returns a vector of shape (6,) with first three elements are position, and last three elements are orientation vector
def FK_combined_orientaion_position(Q, DH_table):

    return np.concatenate((FK_position(Q, DH_table), FK_orientation(Q, DH_table)), axis = None)

# get the coordinates of every joint in the armature given a state vector Q. For plotting purposes
def get_Joint_Positions(Q, DH_table):

    jointPositions_ = np.zeros((len(Q)+1, 3))

    for i in range(len(Q)):

        jointPositions_[i+1] = Combined_Csys_Transform(DH_table(Q)[:i+1])[:3, 3]

    return jointPositions_


# define jacobian matrix for the forward kinematics function f: Q -> X using middle point first order derivative

def Jacobian(f, Q_, DH_table, epsilon = 1e-4):

    # inputs the FK-transformation and the current state Q
    # epsilon is how long step is used to estimate the linear approximation

    q = np.array(Q_, dtype = float)
    n = len(q)
    X = f(q, DH_table)
    m = len(X)
    J = np.zeros((m, n))

    for i in range(n):

        Qplus = q.copy()
        Qminus = q.copy()
        Qplus[i] += epsilon
        Qminus[i] -= epsilon
        Xplus = f(Qplus, DH_table)
        Xminus = f(Qminus, DH_table)

        J[:, i] = (np.array(Xplus) - np.array(Xminus))/(2*epsilon)

    return J

# get condition number of the Jacobian to determine if damping is necessary # - Not used bc DLS makes this kinda redundant
def kappa(A):

    return np.linalg.cond(A)

# Damped Least Squares for use when the Jacobian is near singular, but is always applicable. When the transformation nears a singularity,
# spread the displacement from the most significant joints to more of the redundant joints that have much smaller eigenvalues by setting a minimum (sigma + lambda)
def DLS_JacobianInverse(Jacobian_, l = 1e-4):

    n, m = Jacobian_.shape
    I = np.eye(m)
    return np.linalg.inv(Jacobian_.T @ Jacobian_ + (l**2)*I) @ Jacobian_.T


# define functions to integrate towards target point

# single step
def IK_step(heading_, h_, Q_, DH_table):
    # inputs heading vector, and desired step length h_, and state vector Q_
    # returns the Q-step vector

    return DLS_JacobianInverse(Jacobian(FK_combined_orientaion_position, Q_, DH_table)) @ (h_ * heading_)


# move towards point
def IK_move(Q_, DH_table, target_X_, tol, targetSteps=100, maxSteps=500, alpha=0.8):
    # inputs the current state in Q-space, the target in X-space, tolerance of proximity, target number of steps, max number of steps, and alpha
    # (1.0 > alpha > 0.5) ---> start fast, stop slow;  alpha = 0.5 ---> constant step length). For example, if alpha = 0.70, then 70% of the distance is covered by the first half of the steps, and the remaining 30% distance is covered by the remaining target steps

    Q = np.zeros((maxSteps, len(Q_)))
    Q[0] = Q_
    X = np.zeros((maxSteps, len(target_X_)))
    X[0] = FK_combined_orientaion_position(Q_, DH_table)
    heading = target_X_ - X[0]

    initDistance = np.linalg.norm(heading.copy())

    step = 0
    while np.linalg.norm(heading) > tol and step < maxSteps - 1:

        if np.linalg.norm(heading) > (1 - alpha) * initDistance:  # if still far away from target (defined by alpha)

            Q[step + 1] = Q[step] + IK_step(heading, 2 * alpha * initDistance / targetSteps, Q[step], DH_table)

        else:  # if closing in on target

            Q[step + 1] = Q[step] + IK_step(heading, 2 * (1 - alpha) * initDistance / targetSteps, Q[step], DH_table)

        X[step + 1] = FK_combined_orientaion_position(Q[step + 1], DH_table)

        heading = target_X_ - X[step + 1]
        step += 1

    # outputs both Q-and X trajectories, in addition to the final distance to the desired destination
    return Q[:step], X[:step], np.linalg.norm(heading)




# Define the allowed robot arm constraints here ------------------------------------------------------------------------------------------------

# Define kinematics through the Denavit-Hartenberg table here. (gamma 1, gamma2, d3). Edit the output such that you get the correct table with the constant and variable parameters
def DH_table(Q):

    # Inputs Q, a vector with N_DOF parameters (angles and telescopes): Outputs the finished DH-table

    return [[np.pi + Q[0], np.pi/2, -0.5, 1.045],
            [np.pi/2 + Q[1], 0, -1.3, 0],
            [Q[2], -np.pi/2, 0.055, 0],
            [Q[3], np.pi/2, 0, 1.025],
            [Q[4], -np.pi/2, 0, 0],
            [Q[5], 0, 0, 0.23]
            ]


target_point = np.concatenate([query_point, np.array([0, 0, 1])])
start_angles = np.concatenate([nearest_angles, np.array([0])])

start_jac_time = time.perf_counter()

Q, X, distance = IK_move(start_angles, DH_table, target_point, tol=1e-4, targetSteps=100, maxSteps=500, alpha=0.5)

final_angles = np.array(Q[-1])

end_jac_time = time.perf_counter()

# Compute additional time values
math_time = (end_jac_time - start_jac_time) * 1000
total_time = (end_jac_time - start_time) * 1000

# Calculate timing values
math_time = (end_jac_time - start_jac_time) * 1000
total_time = (end_jac_time - start_time) * 1000

# Print header
print("\n" + "═"*60)
print("        Inverse Kinematics with NNS Summary")
print("═"*60)

# Print fields with consistent spacing
print(f"{'Query Position:':25} {np.array2string(query_point, precision=4)}")
print(f"{'Nearest Point:':25} {nearest_point}")
print(f"{'Nearest Joint Angles:':25} {nearest_angles}")
print(f"{'Final Joint Angles:':25} {np.array2string(final_angles, precision=4)}")
print(f"{'Distance:':25} {dist*100:.4f} cm")
print(f"{'Query Execution Time:':25} {elapsed_time:.4f} μs")
print(f"{'Inverse Jacobian Time:':25} {math_time:.4f} ms")
print(f"{'Total Elapsed Time:':25} {total_time:.4f} ms")
print("═"*60)


total_distances = []
query_times = []
ik_times = []
total_times = []
N = 1000  # number of runs



for i in range(N):


    # Warm-up call to the tree (ignore result)
    _ = kdtree.query(random_query_point(0.1, positions))
    _ = kdtree.query(random_query_point(0.1, positions))
    _ = kdtree.query(random_query_point(0.1, positions))
    _ = kdtree.query(random_query_point(0.1, positions))
    _ = kdtree.query(random_query_point(0.1, positions))

    # Generate random query
    query_point = random_query_point(0.1, positions)

    # Start timer
    start_time = time.perf_counter()

    # KDTree nearest neighbor
    dist, index = kdtree.query(query_point)
    mid_time = time.perf_counter()

    # Prepare IK target and start angles
    nearest_angles = angles[index]
    target_point = np.concatenate([query_point, np.array([0, 0, 1])])
    start_angles = np.concatenate([nearest_angles, np.array([0])])

    prep_time_end = time.perf_counter()

    # Run IK solver
    Q, X, final_dist = IK_move(start_angles, DH_table, target_point, tol=1e-4)
    end_time = time.perf_counter()

    # Record times
    prep_time = prep_time_end - mid_time

    query_times.append((mid_time - start_time) * 1e6)   # microseconds
    ik_times.append(((end_time - mid_time)-prep_time) * 1e3)        # milliseconds
    total_times.append(((end_time - start_time)-prep_time) * 1e3)    # milliseconds

    total_distances.append(dist*100)

# Compute averages
avg_query_time = np.mean(query_times)
avg_ik_time = np.mean(ik_times)
avg_total_time = np.mean(total_times)
avg_total_distance = np.mean(total_distances)


print("        Average IK + NNS Times (1000 runs)")
print("═" * 60)
print(f"{'Average Distance:':25} {avg_total_distance:.4f} cm")
print(f"{'Average Query Time:':25} {avg_query_time:.4f} μs")
print(f"{'Average IK Time:':25} {avg_ik_time:.4f} ms")
print(f"{'Average Total Time:':25} {avg_total_time:.4f} ms")
print("═" * 60)


import matplotlib.pyplot as plt
import numpy as np

# Convert lists to NumPy arrays for easier manipulation
total_times_arr = np.array(total_times)
total_distances_arr = np.array(total_distances)

# Calculate means and standard deviations
mean_time = np.mean(total_times_arr)
std_time = np.std(total_times_arr)
mean_distance = np.mean(total_distances_arr)
std_distance = np.std(total_distances_arr)

# Define the number of standard deviations for the cutoff (3 is a common choice)
cutoff = 3

# Create a boolean mask to keep only those points that lie within the cutoff for both axes
mask = ((np.abs(total_times_arr - mean_time) < cutoff * std_time) &
        (np.abs(total_distances_arr - mean_distance) < cutoff * std_distance))

# Filter the arrays to include only inliers
times_inliers = total_times_arr[mask]
distances_inliers = total_distances_arr[mask]

# Optionally, you can print how many points were removed
num_removed = len(total_times_arr) - len(times_inliers)
print(f"Removed {num_removed} outlier points out of {len(total_times_arr)} total runs.")

# Create a scatter plot of the inlier data
plt.figure(figsize=(8, 6))
plt.scatter(times_inliers, distances_inliers, alpha=0.6, marker='o', color='blue')

# Label the axes and add a title and grid
plt.xlabel("Total Time (ms)")
plt.ylabel("Distance before IK (cm)")
plt.title("1000 Runs: Total Time vs. Distance before IK (Outliers Removed)")
plt.grid(True)

# Optionally, save the figure:
# plt.savefig("time_vs_distance_inliers.png", dpi=300)

plt.show()