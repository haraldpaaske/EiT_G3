import numpy as np
import sympy as sm
from sympy import cos, sin, sqrt, atan2




#Define qq matrix
theta, alpha, a, d = sm.symbols('θ α a d')

T = sm.Matrix([[cos(theta), -sin(theta)*cos(theta),  sin(theta)*sin(alpha), a*cos(theta)],
               [sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
               [    0     ,          sin(alpha)   ,           cos(alpha)  ,        d        ],
               [    0     ,            0          ,                 0     ,        1        ]])

params = sm.Matrix([theta, alpha, a, d])
T_i_i1 = sm.lambdify((params,), T, modules='numpy')


#Set parametres for the robotarm
θ_i = ['t1','t2','t3','t4','t5','t6']


#Calculate xyz position of end effector. 
def forward_6dof(θ, α, r, d):
    
    num_samples = θ.shape[0]
    out = []
    for j in range(num_samples):
        # print(f'j: {j}')
        theta = θ[j]
        T=[]
        for i in range(9):
            # print(f'i: {i}')
            # print(f'th: {theta} \n a: {α} \n r: {r} \n d: {d}')

            params = [theta[i], α[i], r[i], d[i]]
            T.append(T_i_i1(params))


        T_1_6 = T[0]@T[1]@T[2]@T[3]@T[4]@T[5]@T[6]@T[7]@T[8]

        R = T_1_6[:3, :3]     #Rotation matrix from base to end effector
        t = T_1_6[:3, 3]      #Translation from base to end effector

        x = t[0]
        y = t[1]
        z = t[2]

        phi = atan2(R[2,1],R[2,2])
        theta = atan2(-R[2,0], sqrt(R[2,1]**2+R[2,2]**2))
        psi = atan2(R[1,0], R[0,0])
        out.append([x,y,z, phi, theta, psi])

    return out


