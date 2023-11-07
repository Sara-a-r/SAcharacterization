import os
import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt
import scipy.signal
from scipy import signal
from scipy.optimize import root

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

#-----------------------Transfer Function----------------------#
def Matrices(M1, M2, M3, M4, M5, K1, K2, K3, K4, K5, g2, g3, g4, g5):
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
    D = np.array([[0], [0], [0], [0], [0]])

    return A, B, C, D


if __name__ == '__main__':
    # define the parameters of the system
    gamma = [0.5, 0.5, 0.5, 0.5]              # viscous friction coeff [kg/m*s]
    M = [173, 165, 140, 118, 315]                  # filter mass [Kg]
    K = [ 702.63612454,  859.32565959, 2623.82233935, 1424.4203582,  1968.15021859]                     # spring constant [N/m]
    # compute the transfer function
    A, B, C, D = Matrices(*M, *K, *gamma)

    poles, _ = eig(A)
    print("Poli del sistema:", poles)


    #extract imaginary and real part of poles
    real_parts = np.real(poles)
    imag_parts = np.imag(poles)

    # --------------------------Plot results----------------------#
    #plot poles
    plt.figure()
    plt.title('Poles of the system',size=11)
    plt.xlabel('$\sigma$ (real part)')
    plt.ylabel('$j \omega$ (imaginary part)')
    plt.xlim(-0.007, 0.007)
    plt.ylim(-8, 8)
    plt.grid(True)
    plt.minorticks_on()

    plt.axhline(y=0, linestyle='-', color='black', linewidth=1.1)
    plt.axvline(x=0, linestyle='-', color='black', linewidth=1.1)
    plt.scatter(real_parts, imag_parts, marker='x', color='blue')
    plt.show()

    #save the plot in the results dir
    #out_name = os.path.join(results_dir, "Tf_2M2K.png")
    #plt.savefig(out_name)
    #plt.show()

