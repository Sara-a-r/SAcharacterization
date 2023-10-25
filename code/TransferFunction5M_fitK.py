"""
This code implements the transfer function of a system composed of 5 masses and 5 springs.
On the system acts a viscous friction force with a coefficient gamma.
The spring in contact with the wall is forced by a sinusoidal force F=F0*sin(wt)=K1*(x0).
The system is described in time domain by the equation dx/dt = A*x + B*u, y = C*x + D*u and
the transfer function is given by H(s) = C*(s*Id - A)^(-1) * B + D.
The code returns the transfer function for all the output (in this case X1/X0, X2/X0, X3/X0,
X4/X0, X5/X0).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#from numba import njit

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
#@njit
def TransferFunc (f, M1, M2, M3, M4, M5, K1, K2, K3, K4, K5, g2, g3, g4, g5):
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

    B = np.array((K1 / M1, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    C = np.block([0*Id, Id])

    #initialize the transfer matrix
    H = np.zeros((5, len(f)),dtype = 'complex_') #the matrix has 5 rows (like the number of output)
                                                       #and len(w) columns (all the range of frequencies).
                                                       #In each row there is the Tf of a single output
    for i in range(len(f)):
        H_lenOUT = C @ np.linalg.inv((1j*2*np.pi*f[i])*np.eye(10) - A ) @ B #array, len=number of output, these elements are
                                                                    #the values of the Tf of each output at a given freq
        #store each value of the Tf in the corresponding row of H
        H[0][i] = H_lenOUT[0]
        H[1][i] = H_lenOUT[1]
        H[2][i] = H_lenOUT[2]
        H[3][i] = H_lenOUT[3]
        H[4][i] = H_lenOUT[4]

    return H


if __name__ == '__main__':

    #upload data
    ff1, tf1 = np.loadtxt(os.path.join(data_dir, "Tf 3"), unpack=True)
    ff = ff1[200:900]
    tf = tf1[200:900]

    # create an array of frequencies
    f = np.linspace(0, 1e1, 100000)
    # define the parameters of the system
    gamma = [0, 0, 0, 0]  # viscous friction coeff [kg/m*s]
    M = [173, 165, 140, 118, 315]  # filter mass [Kg]


    def fit_func(f, K1, K2, K3, K4, K5):
        Tf = TransferFunc(f, *M, K1, K2, K3, K4, K5, *gamma)
        H = (np.real(Tf) ** 2 + np.imag(Tf) ** 2) ** 1 / 2
        return H[2]

    init = (301.40719519, 112.19661242, 716.69834327, 283.63249926, 605.79115262)
    #call the minimization routine
    pars, covm = curve_fit(fit_func, ff, tf, init)
    print(pars)

    plt.yscale('log')
    plt.xscale('log')
    plt.plot(f, fit_func(f,*pars))
    plt.plot(ff1, tf1)


    plt.show()

