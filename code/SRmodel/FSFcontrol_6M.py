import os
from scipy import signal
from scipy.linalg import eig
import control as ct
import numpy as np
import matplotlib.pyplot as plt


#--------------------Setup the main directories------------------#
#Define the various directories
script_dir = os.getcwd()                         #define current dir
main_dir = script_dir.split('code')[0]           # go up of two directory
results_dir = os.path.join(main_dir, "figure/SR")   #define results dir
data_dir = os.path.join(main_dir, "data")        #define data dir

if not os.path.exists(results_dir):              #if the directory does not exist create it
    os.mkdir(results_dir)

if not os.path.exists(data_dir):                 #if the directory does not exist create it
    os.mkdir(data_dir)

#-----------------------Transfer Function controlled----------------------#
def StateSpaceMatrix( M1, M2, M3, M4, M5, M6, K1, K2, K3, K4, K5, K6, g2, g3, g4, g5, g6):
    # define the matrices of the system from the state-space equations
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

def Kmatrix(A, B, C, D, dpoles):
    # create the system
    sys = ct.StateSpace(A, B, C, D)

    # define desired poles
    desired_poles = np.array(dpoles)

    # compute the gain
    k = ct.place(A, B, desired_poles)

    return k

def TransferFunc(w, A, B, C, D, k):
    #initialize the transfer matrix
    H = np.zeros((6, len(w)),dtype = 'complex_') #the matrix has 5 rows (like the number of output)
                                                       #and len(w) columns (all the range of frequencies).
                                                       #In each row there is the Tf of a single output
    for i in range(len(w)):
        H_lenOUT = C @ np.linalg.inv((1j*w[i])*np.eye(12) - (A-(B@k))) @ B #array, len=number of output, these elements are
                                                                    #the values of the Tf of each output at a given freq
        H_lenOUT = H_lenOUT.squeeze() # remove empty dimension

        #store each value of the Tf in the corresponding row of H
        H[0][i] = H_lenOUT[0]
        H[1][i] = H_lenOUT[1]
        H[2][i] = H_lenOUT[2]
        H[3][i] = H_lenOUT[3]
        H[4][i] = H_lenOUT[4]
        H[5][i] = H_lenOUT[5]

    #compute poles
    poles, _ = eig((A-(B@k)))

    return H, poles

#--------------------------Time evolution using ARMA model------------------------#
def FSF_AR_model(y, A, B, K, N, r):
    return (A - np.outer(B,K)) @ y + B * N * r  #nd array of y at instant n+1

def matrix(M1, M2, M3, M4, M5, M6, K1, K2, K3, K4, K5, K6, g2, g3, g4, g5, g6, dt):
    # defne the matrices A and B
    Id = np.eye(6)
    V = np.array([[1-(dt*g2/M1), dt*g2/M1, 0, 0, 0, 0],
                       [dt*g2/M2, 1-dt*(g2+g3)/M2, dt*g3/M2, 0, 0, 0],
                       [0, dt*g3/M3, 1-dt*(g3+g4)/M3, dt*g4/M3, 0, 0],
                       [0, 0, dt*g4/M4, 1-dt*(g4+g5)/M4, dt*g5/M4, 0],
                       [0, 0, 0, dt*g5/M5, 1-dt*(g5+g6)/M5, dt*g6/M5],
                       [0, 0, 0, 0, dt*g6/M6, 1-dt*g6/M6]])
    X = dt * np.array([[-(K1+K2)/M1, K2/M1, 0, 0, 0, 0],
                       [K2/M2, -(K2+K3)/M2, K3/M2, 0, 0, 0],
                       [0, K3/M3, -(K3+K4)/M3, K4/M3, 0, 0],
                       [0, 0, K4/M4, -(K4+K5)/M4, K5/M4, 0],
                       [0, 0, 0, K5/M5, -(K5+K6)/M5, K6/M5],
                       [0, 0, 0, 0, K6/M6, -K6/M6]])
    A = np.block([[V, X],
                  [dt*Id, Id]])

    B = np.array((dt*K1 / M1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    return A, B

def evolution(evol_method, Nt_step, dt, physical_params, k, control_params, file_name = None):
    #-----------------Initialize the problem-------------------#
    tmax = dt * Nt_step                                  # total time of simulation
    tt = np.arange(0, tmax, dt)                          # temporal grid
    y0 = np.array((0, 0, 0, 0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6))  # initial condition
    y_t = np.copy(y0)                                    # create a copy to evolve it in time

    #----------------------Time evolution----------------------#
    v1, v2, v3, v4, v5, v6 = [[], [], [], [], [], []]      # initialize list of v values
    x1, x2, x3, x4, x5, x6 = [[], [], [], [], [], []]      # initialize list of x values

    # compute the matrices of the system
    A, B = matrix(*physical_params)

    # temporal evolution when the ext force is applied
    i = 0
    for t in tt:
        i = i + 1
        y_t = evol_method(y_t, A, B, k, *control_params)   # step n+1
        v1.append(y_t[0])
        v2.append(y_t[1])
        v3.append(y_t[2])
        v4.append(y_t[3])
        v5.append(y_t[4])
        v6.append(y_t[5])
        x1.append(y_t[6])
        x2.append(y_t[7])
        x3.append(y_t[8])
        x4.append(y_t[9])
        x5.append(y_t[10])
        x6.append(y_t[11])

    # save simulation's data (if it's necessary)
    if file_name is not None:
        data = np.column_stack((tt, v1, v2, v3, v4, v5, v6, x1, x2, x3, x4, x5, x6))
        np.savetxt(os.path.join(data_dir, file_name), data, header='time, v1, v2, v3, v4, v5, v6, x1, x2, x3, x4, x5, x6')

    return (tt, np.array(v1), np.array(v2), np.array(v3), np.array(v4), np.array(v5), np.array(v6),
            np.array(x1), np.array(x2), np.array(x3), np.array(x4), np.array(x5), np.array(x6))




if __name__ == '__main__':
    # create an array of frequencies
    f = np.linspace(1e-2,1e1,10000)
    w = 2*np.pi*f

    freq = np.loadtxt('../../data/freq.txt', unpack=True)   #this is necessary when we compute the product between
    wn = 2*np.pi*freq                                              #ASD of seismometer and TF (TF must be evaluated in the same freq)

    # Parameters of the simulation
    Nt_step = 5e5     #temporal steps
    dt = 1e-3         #temporal step size

    # Parameters of the system
    gamma = [5, 5, 5, 5, 5]                    # viscous friction coeff [kg/m*s]
    M = [160, 125, 120, 110, 325, 82]          # filter mass [Kg]  [M1, M2, M3, M4, M7, Mpayload]
    K = [700, 1500, 3300, 1500, 3400, 564]     # spring constant [N/m]  [K1, K2, K3, K4, K5, K6]

    # control parameters
    r = 0  # theta ref
    N = 112  # factor for scaling the input

    #define the desired poles
    dpoles = [-1.9302313 + 8.44834539j, -1.9302313 - 8.44834539j,   #1.344Hz
              -0.8730504 + 7.11085663j, -0.8730504 - 7.11085663j,   #1.131Hz
              -2.9692883 + 4.39542468j, -2.9692883 - 4.39542468j,   #0.699Hz
              -0.759586  + 0.66927877j, -0.759586  - 0.66927877j,   #0.106Hz
              -1.99151   + 2.24037402j, -1.99151   - 2.24037402j,   #0.356Hz
              -0.7315067 + 3.00566475j, -0.7315067 - 3.00566475j]   #0.478Hz

    # compute the state space matrices
    A, B, C, D = StateSpaceMatrix(*M, *K, *gamma)
    # compute the gain matrix k
    k = Kmatrix(A, B, C, D, dpoles)     #k is a (1,n) matrix (in this case (1, 12) matrix, i.e. row vector)

    # Simulation
    physical_params = [*M, *K, *gamma, dt]
    control_params = [N, r]
    simulation_params = [FSF_AR_model, Nt_step, dt]
    tt, v1, v2, v3, v4, v5, v6, x1, x2, x3, x4, x5, x6 = evolution(*simulation_params,
                                        physical_params, k, control_params, file_name = None)


    # compute the transfer function
    Tf, poles = TransferFunc(wn, A, B, C, D, k)
    # compute the magnitude of the transfer function
    H = (np.real(Tf) ** 2 + np.imag(Tf) ** 2) ** (1 / 2)


    #save H values in a file
    np.savetxt(os.path.join(data_dir, 'TF_FSFcontrol.txt'), np.column_stack((freq, H[0], H[5])), header='f[Hz], H(x1/x0), H(xpl/x0)')

    # extract imaginary and real part of poles
    real_parts = np.real(poles)
    imag_parts = np.imag(poles)

    print("Poles: ", poles)
    print('Real part is: sigma = ', real_parts)
    print('Imaginary part is: w = ', imag_parts)
    print('Normal frequencies are:', (imag_parts[imag_parts>0] / (2 * np.pi)))

    #-----------------------------plot poles-------------------------#
    #plt.figure(figsize=(5, 4))
    plt.title('Poles of the system', size=11)
    plt.xlabel('$\sigma$ (real part)')
    plt.ylabel('$j \omega$ (imaginary part)')
    plt.grid(True)
    plt.minorticks_on()

    plt.axhline(y=0, linestyle='-', color='black', linewidth=1.1)
    plt.axvline(x=0, linestyle='-', color='black', linewidth=1.1)
    plt.scatter(real_parts, imag_parts, marker='x', color='blue', linewidths=0.9)
    plt.show()

    # ----------------------------Plot TF----------------------------#
    #load data TF not controlled
    _, Tfnc_1, Tfnc_pl = np.loadtxt('../../data/TFnoControl.txt',unpack=True)

    plt.title('Transfer function (FSF control)', size=11)
    plt.xlabel('f [Hz]')
    plt.ylabel('|x$_{out}$/x$_0$|')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True)
    plt.minorticks_on()

    plt.plot(freq, Tfnc_pl, linestyle='-', linewidth=1, marker='', color='red', label='no control')
    #plt.plot(freq, H[0], linestyle='-', linewidth=1, marker='', color='steelblue', label='FSF control')
    plt.plot(freq, H[5], linestyle='-', linewidth=1, marker='', color='steelblue', label='x$_{out}$ = x$_{pl}$')
    plt.legend()

    plt.show()

    # -----------------------Plot time evolution-------------------#
    # fig = plt.figure(figsize=(12,10))
    plt.title('Time evolution for SR (FSF control)')
    plt.xlabel('Time [s]')
    plt.ylabel('x [m]')
    plt.grid(True)
    plt.minorticks_on()

    plt.plot(tt, x1, linestyle='-', linewidth=1, marker='', color='steelblue', label='x1, M$_1$')
    plt.plot(tt, x2, linestyle='-', linewidth=1, marker='', color='black', label='x2, M$_2$')
    plt.plot(tt, x3, linestyle='-', linewidth=1, marker='', color='red', label='x3, M$_3$')
    plt.plot(tt, x4, linestyle='-', linewidth=1, marker='', color='green', label='x4, M$_4$')
    plt.plot(tt, x5, linestyle='-', linewidth=1, marker='', color='darkmagenta', label='x7, M$_7$')
    plt.plot(tt, x6, linestyle='-', linewidth=1, marker='', color='pink', label='x$_{pl}$, M$_{pl}$')
    plt.legend()

    plt.show()
