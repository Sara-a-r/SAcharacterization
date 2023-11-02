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
def matrices (M1, M2, M3, M4, M5, K1, K2, K3, K4, K5, g2, g3, g4, g5):
    #define the matrices of the system from the state-space equations
    Id = np.eye(5)
    V = np.array([[-(g2 / M1), g2 / M1, 0, 0, 0],
                  [g2 / M2, -(g2 + g3) / M2, g3 / M2, 0, 0],
                  [0, g3 / M3, -(g3 + g4) / M3, g4 / M3, 0],
                  [0, 0, g4 / M4, -(g4 + g5) / M4, g5 / M4],
                  [0, 0, 0, g5 / M5, -g5 / M5]])
    X = np.array([[-(K1 + K2) / M1, K2 / M1, 0, 0, 0],
                  [K1 / M2, -(K2 + K3) / M2, K3 / M2, 0, 0],
                  [0, K3 / M3, -(K3 + K4) / M3, K4 / M3, 0],
                  [0, 0, K4 / M4, -(K4 + K5) / M4, K5 / M4],
                  [0, 0, 0, K5 / M5, -K5 / M5]])
    A = np.block([[V, X],
                  [Id, 0 * Id]])

    B = np.array([[K1 / M1], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
    C = np.block([0*Id, Id])
    D = np.zeros((5, 1))

    return A, B, C, D

#-------------------------------Control Matrix--------------------------#
def ctrl_martix(A, B):
    n = A.shape[0]
    Ctrl = B
    for i in range(1,n):
        Ctrl = np.hstack((Ctrl, np.linalg.matrix_power(A,i) @ B))

    return Ctrl

if __name__ == '__main__':
    # define the parameters of the system
    gamma = [0.5, 0.5, 0.5, 0.5]  # viscous friction coeff [kg/m*s]
    M = [173, 165, 140, 118, 315]  # filter mass [Kg]
    K = [240.1762472, 1591.49007496, 1765.26873492, 309.85508443, 3920.7499088]  # spring constant [N/m]

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

