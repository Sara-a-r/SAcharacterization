import os
import numpy as np
import matplotlib.pyplot as plt


#--------------------Setup the main directories------------------#
#Define the various directories
script_dir = os.getcwd()                         #define current dir
main_dir = os.path.dirname(script_dir)           #go up of one directory
results_dir = os.path.join(main_dir, "figure")   #define results dir
data_dir = os.path.join(main_dir, "data")        #define data dir

if not os.path.exists(results_dir):              #if the directory does not exist create it
    os.mkdir(results_dir)

if not os.path.exists(data_dir):                 #if the directory does not exist create it
    os.mkdir(data_dir)

#-----------------------state space description----------------------#
#@njit
def matrices (M1, M2, M3, M4, M5, M6, K1, K2, K3, K4, K5, K6, g2, g3, g4, g5, g6):
    #define the matrices of the system from the state-space equations
    Id = np.eye(6)
    V = np.array([[-(g2 / M1), g2 / M1, 0, 0, 0, 0],
                  [g2 / M2, -(g2 + g3) / M2, g3 / M2, 0, 0, 0],
                  [0, g3 / M3, -(g3 + g4) / M3, g4 / M3, 0, 0],
                  [0, 0, g4 / M4, -(g4 + g5) / M4, g5 / M4, 0],
                  [0, 0, 0, g5 / M5, -(g5 + g6) / M5, g6 / M5],
                  [0, 0, 0, 0, g6 / M6, -g6 / M6]])
    X = np.array([[-(K1 + K2) / M1, K2 / M1, 0, 0, 0, 0],
                  [K2 / M2, -(K2 + K3) / M2, K3 / M2, 0, 0, 0],
                  [0, K3 / M3, -(K3 + K4) / M3, K4 / M3, 0, 0],
                  [0, 0, K4 / M4, -(K4 + K5) / M4, K5 / M4, 0],
                  [0, 0, 0, K5 / M5, -(K5 + K6) / M5, K6 / M5],
                  [0, 0, 0, 0, K6 / M6, -K6 / M6]])
    A = np.block([[V, X],
                  [Id, 0 * Id]])

    B = np.array([[K1 / M1], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
    C = np.block([0 * Id, Id])
    D = np.array([[0], [0], [0], [0], [0], [0]])

    return A, B, C, D

#---------------------------Controllability Matrix------------------------#
def ctrl_martix(A, B):
    n = A.shape[0]
    Ctrl = B
    for i in range(1,n):
        Ctrl = np.hstack((Ctrl, np.linalg.matrix_power(A,i) @ B))

    return Ctrl

#--------------------Control Canonical Form---------------------#
def matrices_c(Ctrl, A, B, C, D):
    n = A.shape[0]
    # compute the transformation matrix
    t_n = np.array([0,0,0,0,0,0,0,0,0,0,0,1]) @ np.linalg.inv(Ctrl)
    T_inv = t_n @ np.linalg.matrix_power(A,n-1)    #first row of T_inv matrix
    for i in range(2,n+1):
        T_inv = np.vstack((T_inv,t_n @ np.linalg.matrix_power(A,n-i)))

    # compute the matrices in control canonical form
    Ac = T_inv @ A @ np.linalg.inv(T_inv)
    Bc = T_inv @ B
    Cc = C @ np.linalg.inv(T_inv)
    Dc = D
    return Ac, Bc, Cc, Dc


if __name__ == '__main__':
    # define the parameters of the system
    gamma = [5, 5, 5, 5,5]  # viscous friction coeff [kg/m*s]
    M = [160, 125, 120, 110, 325, 82]  # filter mass [Kg]  [M1, M2, M3, M4, M7, Mpayload]
    K = [700, 1500, 3300, 1500, 3400, 564]  # spring constant [N/m]  [K1, K2, K3, K4, K5, K6]

    # compute the state space matrices
    A, B, C, D = matrices(*M, *K, *gamma)

    # compute controllability matrix
    Ctrl = ctrl_martix(A,B)
    # verify if the system is controllable
    Ctrl_rank = np.linalg.matrix_rank(Ctrl)
    if Ctrl_rank == A.shape[0]:
        print("The system is controllable.")
    else:
        print("The system is not fully controllable.")

    # compute matrices in control canonical form
    Ac, Bc, Cc, Dc = matrices_c(Ctrl, A, B, C, D)

    print("A_c = \n", Ac)
    print("B_c = \n", Bc)
    print("C_c = \n", Cc)
    print("D_c = \n", Dc)