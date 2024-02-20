import os
import numpy as np
import cmath
import matplotlib.pyplot as plt
from scipy import signal

#--------------------Setup the main directories------------------#
#Define the various directories
script_dir = os.getcwd()                         #define current dir
main_dir = script_dir.split('code')[0]           # go up of two directory
results_dir = os.path.join(main_dir, "figure")   #define results dir
data_dir = os.path.join(main_dir, "data")        #define data dir

if not os.path.exists(results_dir):              #if the directory does not exist create it
    os.mkdir(results_dir)

if not os.path.exists(data_dir):                 #if the directory does not exist create it
    os.mkdir(data_dir)

#--------------------------AR model------------------------#
def AR_model(y, A, B, u):
    return A @ y + B * u  #nd array of y at instant n+1

#--------------------------Right Hand Side------------------------#
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

    B = np.array((dt/M1,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    return A, B

#--------------------------Temporal evolution----------------------#
def evolution(evol_method, Nt_step, dt, physical_params, ref, kp, ki, kd, file_name = None):
    #-----------------Initialize the problem-------------------#
    tmax = dt * Nt_step                                 #total time of simulation
    tt = np.arange(0, tmax, dt)                         #temporal grid
    y0 = np.array((0, 0, 0, 0, 0,0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6))       #initial condition
    y_t = np.copy(y0)                                   #create a copy to evolve it in time

    #----------------------Time evolution----------------------#
    v1, v2, v3, v4, v5, v6 = [[], [], [], [], [], []]  # initialize list of v values
    x1, x2, x3, x4, x5, x6 = [[], [], [], [], [], []]  # initialize list of x values

    #compute the matrices of the system
    A, B = matrix(*physical_params)

    #parameters for PID control
    err_t = []  # list for memorizing the value of the error
    Fc = 0       # PID term
    I = 0
    j = 0       #cycle index

    #temporal evolution when the ext (control) force is applied
    for t in tt:
        y_t = evol_method(y_t, A, B, Fc)   #step n+1
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

        err = ref - y_t[6]                      # evaluate the error
        err_t.append(err)
        delta_err = err - err_t[j - 1]
        P = kp * err                            # calculate P term
        I = I + ki * (err * dt)                 # calculate the I term
        D = kd * (delta_err / dt)               # calculate the D term
        Fc = P + I + D                          # calculate PID term
        j = j + 1

    #save simulation's data (if it's necessary)
    if file_name is not None:
        data = np.column_stack((tt, v1, v2, v3, v4, v5, v6, x1, x2, x3, x4, x5, x6))
        np.savetxt(os.path.join(data_dir, file_name), data, header='time, v1, v2, v3, v4, v5, v6, x1, x2, x3, x4, x5, x6')

    return (tt, np.array(v1), np.array(v2), np.array(v3), np.array(v4), np.array(v5), np.array(v6),
            np.array(x1), np.array(x2), np.array(x3), np.array(x4), np.array(x5), np.array(x6))

#-----------------------Plant Transfer Function----------------------#
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
    A = np.block([[V, X],
                  [Id, 0 * Id]])

    B = np.array((K1 / M1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    C = np.block([0*Id, Id])

    #initialize the transfer matrix
    H = np.zeros((6, len(w)),dtype = 'complex_') #the matrix has 5 rows (like the number of output)
                                                       #and len(w) columns (all the range of frequencies).
                                                       #In each row there is the Tf of a single output
    for i in range(len(w)):
        H_lenOUT = C @ np.linalg.inv((1j*w[i])*np.eye(12) - A) @ B #array, len=number of output, these elements are
                                                                    #the values of the Tf of each output at a given freq
        #store each value of the Tf in the corresponding row of H
        H[0][i] = H_lenOUT[0]
        H[1][i] = H_lenOUT[1]
        H[2][i] = H_lenOUT[2]
        H[3][i] = H_lenOUT[3]
        H[4][i] = H_lenOUT[4]
        H[5][i] = H_lenOUT[5]
    return H

#---------------------Controller Transfer function---------------------#
def Controller_Tf(w, kp, ki, kd):
    s = 1j * w

    # Create/view notch filter
    samp_freq = 1000  # Sample frequency (Hz)
    notch_freq = 0.106  # Frequency to be removed from signal (Hz)
    quality_factor = 0.01  # Quality factor

    # Design a notch filter using signal.iirnotch
    b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, samp_freq)

    # Compute magnitude response of the designed filter
    freq, h = signal.freqz(b_notch, a_notch, worN=np.linspace(1e-2, 1e1, 10000), fs=1000)

    return ( kp + (ki/s) + (kd * s) ) * h


if __name__ == '__main__':

    #Parameters of the simulation
    Nt_step = 5e5     #temporal steps
    dt = 1e-3         #temporal step size

    # create an array of frequencies
    f = np.linspace(1e-2, 1e1, 10000) #step=0.1Hz
    w = 2 * np.pi * f

    #Parameters of the system
    gamma = [5, 5, 5, 5, 5]  # viscous friction coeff [kg/m*s]
    M = [160, 125, 120, 110, 325, 82]  # filter mass [Kg]  [M1, M2, M3, M4, M7, Mpayload]
    K = [700, 1500, 3300, 1500, 3400, 564]  # spring constant [N/m]  [K1, K2, K3, K4, K5, K6]

    #Parameters of control
    ref = 0                                     # reference signal for x1
    kp = 0.8
    ki = 0.3
    kd = 0.5
    control_params_PID = [ref, kp, ki, kd]

    #Simulation
    physical_params = [*M, *K, *gamma, dt]
    simulation_params = [AR_model, Nt_step, dt]
    tt, v1_PID, v2_PID, v3_PID, v4_PID, v5_PID, v6_PID, x1_PID, x2_PID, x3_PID, x4_PID, x5_PID, x6_PID = evolution(*simulation_params,
                                                           physical_params, *control_params_PID,
                                                           file_name=None)

    #-----------------------Transfer function---------------------#
    H = TransferFunc(w, *M, *K, *gamma)
    C = Controller_Tf(w, kp, ki, kd)

    Tf = ((np.real(H)) ** 2 + (np.imag(H)) ** 2) ** (1 / 2)     #plant Tf

    G_ol = H * C
    Tf_ol = ((np.real(G_ol)) ** 2 + (np.imag(G_ol)) ** 2) ** (1 / 2)

    G_cl = G_ol / (1 + G_ol)
    Tf_cl = ((np.real(G_cl)) ** 2 + (np.imag(G_cl)) ** 2) ** (1 / 2)

    #evaluate poles and zeros
    zeros_p = (- kp + cmath.sqrt(kp ** 2 - (4 * kd * ki))) / (2 * kd)
    zeros_m = (- kp - cmath.sqrt(kp ** 2 - (4 * kd * ki))) / (2 * kd)
    zeros_p_real = np.real(zeros_p)
    zeros_p_img = np.imag(zeros_p)
    zeros_m_real = np.real(zeros_m)
    zeros_m_img = np.imag(zeros_m)

    print('PID controller\'s poles and zeros are:')
    if (ki != 0) : print('poles = 0')
    print('real zeros are: %.4f, %.4f' % (zeros_p_real, zeros_m_real))
    print('imag zeros are: %.4f, %.4f' % (zeros_p_img, zeros_m_img))


    # --------------------------Plot results----------------------#

    #Transfer function
    plt.title('Transfer function SR controlled')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('|x$_{out}$/x$_0$|')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True)
    plt.minorticks_on()

    plt.plot(f, Tf[0], linestyle=':', linewidth=1, marker='', color='gray', label='output $x_1$')
    plt.plot(f, Tf_ol[0], linestyle='-', linewidth=1, marker='', color='red', label='open loop Tf')
    plt.plot(f, Tf_cl[0], linestyle='-', linewidth=1, marker='', color='green', label='close loop Tf')
    plt.plot(f, np.sqrt((np.real(C))**2+(np.imag(C))**2), linestyle='-', linewidth=1, marker='', color='pink', label='output $x_1$')
    plt.legend()

    plt.show()

    #Time evolution
    #fig = plt.figure(figsize=(12,10))
    #plt.title('PID control for five coupled oscillators \n x$_1^{ref}$=%.1f, k$_p$=%.1f, k$_i$=%.1f, k$_d$=%.1f'
    #          %(control_params_PID[0], control_params_PID[1], control_params_PID[2], control_params_PID[3]), size=11)
    plt.title('Time evolution for six coupled oscillators')
    plt.xlabel('Time [s]')
    plt.ylabel('position [m]')
    plt.grid(True)
    plt.minorticks_on()

    plt.plot(tt, x1_PID, linestyle='-', linewidth=1, marker='', color='steelblue', label='x1, mass M1')
    #plt.plot(tt, x2_PID, linestyle='-', linewidth=1, marker='', color='black', label='x2, mass M2')
    #plt.plot(tt, x3_PID, linestyle='-', linewidth=1, marker='', color='red', label='x3, mass M3')
    #plt.plot(tt, x4_PID, linestyle='-', linewidth=1, marker='', color='green', label='x4, mass M4')
    #plt.plot(tt, x5_PID, linestyle='-', linewidth=1, marker='', color='darkmagenta', label='x5, mass M7')
    #plt.plot(tt, x6_PID, linestyle='-', linewidth=1, marker='', color='pink', label='x6, mass M$_{pl}$')
    plt.legend()


    #save the plot in the results dir
    #out_name = os.path.join(results_dir, "SinResp_2M.png")
    #plt.savefig(out_name)
    plt.show()

