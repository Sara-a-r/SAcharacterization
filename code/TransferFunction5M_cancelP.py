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
def TransferFunc (w, M1, M2, M3, M4, M5, K1, K2, K3, K4, K5, g2, g3, g4, g5):
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
    H = np.zeros((5, len(w)),dtype = 'complex_') #the matrix has 5 rows (like the number of output)
                                                       #and len(w) columns (all the range of frequencies).

                                                 #In each row there is the Tf of a single output
    for i in range(len(w)):
        H_lenOUT = C @ np.linalg.inv((1j*w[i])*np.eye(10) - A ) @ B #array, len=number of output, these elements are
                                                                    #the values of the Tf of each output at a given freq


        #store each value of the Tf in the corresponding row of H
        H[0][i] = H_lenOUT[0]
        H[1][i] = H_lenOUT[1]
        H[2][i] = H_lenOUT[2]
        H[3][i] = H_lenOUT[3]
        H[4][i] = H_lenOUT[4]
    return H

#----------------------------Phase-----------------------------#
def Phase(Tf):
    return np.arctan(np.imag(Tf)/np.real(Tf)) * 180 /np.pi

#-------------------------Bode plot-----------------------------#
def Bode(Tf):
    return 20 * np.log10(np.abs(Tf))

if __name__ == '__main__':
    # create an array of frequencies
    f = np.linspace(0,1e1,100000)
    w = 2*np.pi*f
    # define the parameters of the system
    gamma = [0.5, 0.5, 0.5, 0.5]              # viscous friction coeff [kg/m*s]
    M = [173, 165, 140, 118, 315]                  # filter mass [Kg]
    #K = [ 240.1762472, 1591.49007496, 1765.26873492,  309.85508443, 3920.7499088 ]                     # spring constant [N/m]
    K = [702.63612454,  859.32565959, 2623.82233935, 1424.4203582,  1968.15021859]
    # compute the transfer function
    Tf = TransferFunc(w, *M, *K, *gamma)
    # compute the magnitude of the transfer function
    H = (np.real(Tf)**2+np.imag(Tf)**2)**(1/2)

    A = Bode(Tf)


    # --------------------------Plot results----------------------#
    #fig = plt.figure(figsize=(10, 7))
    plt.title('Transfer function of coupled oscillators \n M$_1$=%d, M$_2$=%d, M$_3$=%d, M$_4$=%d, M$_5$=%d, K$_1$=%d,'
              'K$_2$=%d, K$_3$=%d, K$_4$=%d, K$_5$=%d, $\gamma$=%.1f' % (M[0],M[1], M[2], M[3], M[4],K[0], K[1], K[2], K[3], K[4],gamma[0]),
              size=11)
    plt.xlabel('f [Hz]')
    plt.ylabel('|x$_{out}$/x$_0$|')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True)
    plt.minorticks_on()

    plt.plot(f, H[0], linestyle='-', linewidth=1, marker='', color='steelblue', label='output $x_1$')
    plt.plot(f, H[1], linestyle='-', linewidth=1, marker='', color='violet', label='output $x_2$')
    plt.plot(f, H[2], linestyle='-', linewidth=1, marker='', color='black', label='output $x_3$')
    plt.plot(f, H[3], linestyle='-', linewidth=1, marker='', color='red', label='output $x_4$')
    plt.plot(f, H[4], linestyle='-', linewidth=1, marker='', color='lime', label='output $x_5$')
    plt.legend()
    plt.tight_layout()

    #save the plot in the results dir
    #out_name = os.path.join(results_dir, "Tf_2M2K.png")
    #plt.savefig(out_name)
    plt.show()

