import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sm
from sympy import cos, sin, sqrt, atan2
import torch
import numpy as np
#Define qq matrix


def kin_plot(theta, goal):
    theta = theta.detach().numpy()
    t_s, a_s, r_s, d_s = sm.symbols('θ α a d')

    T = sm.Matrix([[cos(t_s), -sin(t_s)*cos(t_s),  sin(t_s)*sin(a_s), r_s*cos(t_s)],
               [sin(t_s),  cos(t_s)*cos(a_s), -cos(t_s)*sin(a_s), r_s*sin(t_s)],
               [    0     ,          sin(a_s)   ,           cos(a_s)  ,        d_s        ],
               [    0     ,            0          ,                 0     ,        1        ]])

    params = sm.Matrix([t_s, a_s, r_s, d_s])
    T_i_i1 = sm.lambdify((params,), T, modules='numpy')

    #__________________________________________
    alpha = np.array([0,90,90,0,-90,-90,90,-90,0])
    d= np.array([0,0.479,0.5,0.178,0,0.0557,0.536,0,0.237])
    r = np.array([0.566,-0.067,0,1.3,0.489,0,0,0,0])

    theta = np.array([ 0.3999, -0.2630,  0.5584, -1.3046,  0.4586,  0.2989])

    theta = np.column_stack([
                            0, 
                            theta[0],
                            90,
                            theta[1],
                            90 + theta[2],
                            -90,
                            90 + theta[3],
                            theta[4],
                            theta[5],
                            ])


    params = np.array([theta[0], alpha, r, d])
    params=np.transpose(params)

    points = np.array([[0,0,0]])
    Tt = np.eye(4)
    for par in params:
        Tt = Tt @ T_i_i1(par)
        points = np.vstack((points, Tt[:3,3]))

    #valid: 0, 1, 3, 4, 6, 7, 8
    points = np.delete(points, [2,5], axis=0)


    X, Y, Z = points[:,0], points[:,1], points[:,2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X, Y, Z, '-o', markersize=8, label="Robot Arm")
    ax.scatter(X, Y, Z, color='r', s=50)  # Mark joints
    ax.scatter(goal[0], goal[1], goal[2], color='y', s=100)
    # Label axes
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("3D Robot Arm Visualization")
    ax.legend()
    plt.savefig('marius_template/test_plot/1.png')
    plt.show()
