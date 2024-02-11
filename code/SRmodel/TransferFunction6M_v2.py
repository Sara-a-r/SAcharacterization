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
def TransferFunc (w, M1, M2, M3, M4, M5, M6, K1, K2, K3, K4, K5, K6, g2, g3, g4, g5, g6):
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
                  [0, 0, 0, 0, K6/ M6, -K6 / M6]])

    N = np.array([[1, 0, 0, 0, 0, 0],
                  [-1, 1, 0, 0, 0, 0],
                  [0, -1, 1, 0, 0, 0],
                  [0, 0, -1, 1, 0, 0],
                  [0, 0, 0, -1, 1, 0],
                  [0, 0, 0, 0, 1, 0]])

    A = np.block([[V, X],
                  [Id, 0 * Id]])

    B = np.array((K1 / M1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    C = np.block([0*Id, N])
    D = np.array((-1, 0, 0, 0, 0, -1))


    #initialize the transfer matrix
    H = np.zeros((6, len(w)),dtype = 'complex_') #the matrix has 5 rows (like the number of output)
                                                       #and len(w) columns (all the range of frequencies).
                                                       #In each row there is the Tf of a single output
    for i in range(len(w)):
        H_lenOUT = C @ np.linalg.inv((1j*w[i])*np.eye(12) - A) @ B + D #array, len=number of output, these elements are
                                                                       #the values of the Tf of each output at a given freq
        #store each value of the Tf in the corresponding row of H
        H[0][i] = H_lenOUT[0]
        H[1][i] = H_lenOUT[1]
        H[2][i] = H_lenOUT[2]
        H[3][i] = H_lenOUT[3]
        H[4][i] = H_lenOUT[4]
        H[5][i] = H_lenOUT[5]
    return H

#----------------------------Phase-----------------------------#
def Phase(Tf):
    return np.arctan(np.imag(Tf)/np.real(Tf)) * 180 /np.pi

#-------------------------Bode plot-----------------------------#
def Bode(Tf):
    return 20 * np.log10(np.abs(Tf))

if __name__ == '__main__':
    # create an array of frequencies
    f = np.linspace(1e-2,1e1,10000)
    w = 2*np.pi*f
    # define the parameters of the system
    gamma = [4, 4, 4, 4, 4]              # viscous friction coeff [kg/m*s]
    #M = [160, 155, 135, 128, 400, 125]                  # filter mass [Kg]
    M = [173, 165, 140, 118, 315, 125]
    K = [3923.14475508, 4907.65263311, 1242.11744897, 316.50112348, 574.12017532,
         4522.18941688]  # spring constant [N/m]
    #K = [900, 1900, 3800, 2000, 3700, 875]

    # compute the transfer function
    Tf = TransferFunc(w, *M, *K, *gamma)
    # compute the magnitude of the transfer function
    H = (np.real(Tf)**2+np.imag(Tf)**2)**(1/2)

    A = Bode(Tf)

    #---------------------------Load data-------------------------#
    ff, Tf_m = np.loadtxt('../../data/SR_verticalTF_Ruggi.txt',unpack=True)
    Tf_m = Tf_m * 0.0014  #rescaling factor

    # --------------------------Plot results----------------------#
    #fig = plt.figure(figsize=(10, 7))
    plt.title('Transfer function of SR\n M$_1$=%d, M$_2$=%d, M$_3$=%d, M$_4$=%d, M$_7$=%d, M$_{pl}$=%d, K$_1$=%d,'
              'K$_2$=%d, K$_3$=%d, K$_4$=%d, K$_5$=%d, K$_6$=%d, $\gamma$=%.1f' % (M[0],M[1], M[2], M[3], M[4], M[5], K[0], K[1], K[2],
            K[3], K[4], K[5], gamma[0]),
              size=11)
    plt.xlabel('f [Hz]')
    plt.ylabel('|x$_{out}$/x$_0$|')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True)
    plt.minorticks_on()

    plt.plot(f, H[0], linestyle='-', linewidth=1, marker='', color='steelblue', label='output $x_1$')
    #plt.plot(f, H[1], linestyle='-', linewidth=1, marker='', color='violet', label='output $x_2$')
    #plt.plot(f, H[2], linestyle='-', linewidth=1, marker='', color='black', label='output $x_3$')
    #plt.plot(f, H[3], linestyle='-', linewidth=1, marker='', color='red', label='output $x_4$')
    #plt.plot(f, H[4], linestyle='-', linewidth=1, marker='', color='lime', label='output $x_7$')
    #plt.plot(f, H[5], linestyle='-', linewidth=1, marker='', color='pink', label='output $x_{pl}$')

    plt.plot(ff, Tf_m, linestyle='-', linewidth=1, marker='', color='red', label='open loop data')
    plt.legend()
    plt.tight_layout()

    #save the plot in the results dir
    #out_name = os.path.join(results_dir, "Tf_2M2K.png")
    #plt.savefig(out_name)
    plt.show()

