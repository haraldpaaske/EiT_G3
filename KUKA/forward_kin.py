import numpy as np
import sympy as sm
from sympy import cos, sin

#Set parametres for the robotarm
θ = ['t1','t2','t3','t4','t5','t6']
α = ['a1','a2','a3','a4','a5','a6']
r= ['al1','al2','al3','al4','al5','al6']
d = ['d1','d2','d3','d4','d5','d6']


#Define qq matrix
theta, alpha, a, d = sm.symbols('θ α a d')

T = sm.Matrix([[cos(theta), -sin(theta)*cos(theta),  sin(theta)*sin(alpha), alpha*cos(theta)],
               [sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), alpha*sin(theta)],
               [    0     ,          sin(alpha)   ,           cos(alpha)  ,        d        ],
               [    0     ,            0          ,                 0     ,        1        ]])

params = sm.Matrix([theta, alpha, a, d])
T_i_i1 = sm.lambdify((params,), T, modules='numpy')



#Calculate xyz position of end effector. 
def forward_6dof(θ, α, r, d):
    T=[]
    for i in range(5):
        params = [θ[i], α[i], r[i], d[i]]
        T.append(T_i_i1(params))

    T_1_6 = T[0]@T[1]@T[2]@T[3]@T[4]@T[5]

    x = T_1_6[0,3]
    y = T_1_6[1,3]
    z = T_1_6[2,3]

    return(x,y,z)


