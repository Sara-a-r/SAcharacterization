"""
This code implements the ARMA model for a system composed of 6 masses and 6 springs.
On the system acts a viscous friction force with a coefficient gamma.
The spring in contact with the wall is forced by a sinusoidal force F=F0*sin(wt)=K1*(x0).
The code returns the masses' position and speed in the time domain.

Note = if you need you can save data.
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

#--------------------------AR model------------------------#
def AR_model(y, A, B, u):
    return A @ y + B * u  # nd array of y at instant n+1

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

    B = np.array((dt*K1/M1,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    return A, B
#----------------------------Step function------------------------#
def step_function(t, t0=0):
    return np.heaviside(t-t0, 1)    # 0 if t < t0
                                    # 1 if t >= t0
#----------------------------Sine function------------------------#
def sin_function(t, F0, w):
    return F0 * np.sin(w*t)

#--------------------------Temporal evolution----------------------#
def evolution(evol_method, Nt_step, dt, physical_params, signal_params, F, file_name = None):
    #-----------------Initialize the problem-------------------#
    tmax = dt * Nt_step                                  # total time of simulation
    tt = np.arange(0, tmax, dt)                          # temporal grid
    y0 = np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))  # initial condition
    y_t = np.copy(y0)                                    # create a copy to evolve it in time
    F_signal = F(tt, *signal_params)                     # external force applied to the system in time
    #----------------------------------------------------------#

    #----------------------Time evolution----------------------#
    v1, v2, v3, v4, v5, v6 = [[], [], [], [], [], []]      # initialize list of v values
    x1, x2, x3, x4, x5, x6 = [[], [], [], [], [], []]      # initialize list of x values

    # compute the matrices of the system
    A, B = matrix(*physical_params)

    # temporal evolution when the ext force is applied
    for Fi in F_signal:
        y_t = evol_method(y_t, A, B, Fi)   # step n+1
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
        v6.append(y_t[11])

    # save simulation's data (if it's necessary)
    if file_name is not None:
        data = np.column_stack((tt, v1, v2, v3, v4, v5, v6, x1, x2, x3, x4, x5, x6))
        np.savetxt(os.path.join(data_dir, file_name), data, header='time, v1, v2, v3, v4, v5, v6, x1, x2, x3, x4, x5, x6')

    return (tt, np.array(v1), np.array(v2), np.array(v3), np.array(v4), np.array(v5), np.array(v6),
            np.array(x1), np.array(x2), np.array(x3), np.array(x4), np.array(x5), np.array(x6))


if __name__ == '__main__':

    # Parameters of the simulation
    Nt_step = 1e6     #temporal steps
    dt = 1e-3         #temporal step size

    # Parameters of the system
    gamma = [5, 5, 5, 5, 5]                    # viscous friction coeff [kg/m*s]
    M = [173, 165, 140, 118, 315, 125]                   # filter mass [Kg]  [M1, M2, M3, M4, M7, Mpayload]
    K = [1623.75124242, 3706.96969851, 600.49266591,
         4223.06657828, 1161.6710071, 1598.0074804]      # spring constant [N/m]  [K1, K2, K3, K4, K5, K6]
    t0 = 0                                               # parameter of the step function [s]
    F0 = 1                                               # amplitude of the external force #Note : consider also the coeff in B
    w = 10                                               # f of the ext force

    # Signal applied to the system
    F = sin_function

    # Simulation
    physical_params = [*M, *K, *gamma, dt]
    signal_params = [F0, w]
    simulation_params = [AR_model, Nt_step, dt]
    tt, v1, v2, v3, v4, v5, v6, x1, x2, x3, x4, x5, x6 = evolution(*simulation_params,
                                        physical_params, signal_params, F, file_name = None)


    # --------------------------Plot results----------------------#
    #fig = plt.figure(figsize=(12,10))
    plt.title('Time evolution for SR (AR model)')
    plt.xlabel('Time [s]')
    plt.ylabel('position [m]')
    plt.grid(True)
    plt.minorticks_on()

    plt.plot(tt, x1, linestyle='-', linewidth=1, marker='', color='steelblue', label='x1, mass M1')
    plt.plot(tt, x5, linestyle='-', linewidth=1, marker='', color='darkmagenta', label='x5, mass M5')
    plt.legend()
    plt.tight_layout()

    #save the plot in the results dir
    #out_name = os.path.join(results_dir, "SinResp_5M5Kgamma.png")
    #plt.savefig(out_name)
    plt.show()

