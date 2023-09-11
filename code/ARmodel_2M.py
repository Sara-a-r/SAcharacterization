"""
This code implements the ARMA model for a system constituted by 2 masses M1 and M2
and 2 springs K1 and K2. On the system acts a viscous friction force with a coefficient gamma.
The spring in contact with the wall is forced by a sinusoidal force F=F0*sin(wt).
The code returns the masses' position and speed (x1, x2, v1, v2) in time domain.
"""


import os
import numpy as np
import matplotlib.pyplot as plt

#--------------------Setup the main directories------------------#
#Define the various directories
script_dir = os.getcwd()                         #define current dir
main_dir = os.path.dirname(script_dir)           #go up of one directory
results_dir = os.path.join(main_dir, "figure")   #define results dir

if not os.path.exists(results_dir):              #if the directory does not exist create it
    os.mkdir(results_dir)

#--------------------------AR model------------------------#
def AR_model(y, A, B, u):
    return A @ y + B * u  #nd array of y at instant n+1

#--------------------------Right Hand Side------------------------#
def matrix(gamma, M1, M2, K1, K2, dt):

    #defne the rows of the matrix A
    A_row1 = [1-(2*dt*gamma/M1), dt*gamma/M1, -dt*(K1+K2)/M1, dt*K2/M1]
    A_row2 = [dt*gamma/M2, 1-(dt*gamma/M2), dt*K2/M2, -dt*K2/M2 ]
    A_row3 = [dt, 0, 1, 0]
    A_row4 = [0, dt, 0, 1]

    #define A and B
    A = np.array([A_row1, A_row2, A_row3, A_row4])
    B = np.array((dt/M1,0, 0, 0))
    return A, B
#----------------------------Step function------------------------#
def step_function(t, t0=0):
    return np.heaviside(t-t0, 1)    # 0 if t < t0
                                    # 1 if t >= t0
#----------------------------Sine function------------------------#
def sin_function(t, F0, w):
    return F0 * np.sin(w*t)

#--------------------------Temporal evolution----------------------#
def evolution(evol_method, Nt_step, dt, physical_params, signal_params, F):
    #-----------------Initialize the problem-------------------#
    tmax = dt * Nt_step                #total time of simulation
    tt = np.arange(0, tmax, dt)        #temporal grid
    y0 = np.array((0, 0, 0, 0))        #initial condition
    y_t = np.copy(y0)                  #create a copy to evolve it in time
    F_signal = F(tt, *signal_params)   #external force applied to the system in time
    #----------------------------------------------------------#

    #----------------------Time evolution----------------------#
    v1 = []      #initialize list of v1 value
    v2 = []      # initialize list of v2 value
    x1 = []      #initialize list of x1 value
    x2 = []      # initialize list of x2 value

    #compute the matrices of the system
    A, B = matrix(*physical_params)

    #temporal evolution when the ext force is applied
    for Fi in F_signal:
        y_t = evol_method(y_t, A, B, Fi)   #step n+1
        v1.append(y_t[0])
        v2.append(y_t[1])
        x1.append(y_t[2])
        x2.append(y_t[3])
    return tt, np.array(v1), np.array(v2), np.array(x1), np.array(x2)


if __name__ == '__main__':

    #Parameters of the simulation
    Nt_step = 1e6     #temporal steps
    dt = 1e-2         #temporal step size

    #Parameters of the system
    gamma = 0.2     #viscous friction coeff [kg/m*s]
    M1 = 20         #filter mass [Kg]
    M2 = 20
    K1 = 10         #spring constant [N/m]
    K2 = 10
    t0 = 0          #parameter of the step function [s]
    F0 = 2          #amplitude of the external force
    w = 10          #f of the ext force

    #Signal applied to the system
    F = sin_function

    #Simulation
    physical_params = [gamma, M1, M2, K1, K2, dt]
    signal_params = [F0, w]
    simulation_params = [AR_model, Nt_step, dt]
    tt, v1, v2, x1, x2 = evolution(*simulation_params, physical_params, signal_params, F)

    # --------------------------Plot results----------------------#
    plt.title('Time evolution for two coupled oscillators (AR model)')
    plt.xlabel('Time [s]')
    plt.ylabel('position [m]')
    plt.grid(True)
    plt.minorticks_on()

    plt.plot(tt, x1, linestyle='-', linewidth=1, marker='', color='steelblue', label='x1, mass M1')
    plt.plot(tt, x2, linestyle='-', linewidth=1, marker='', color='darkmagenta', label='x2, mass M2')
    plt.legend()

    #save the plot in the results dir
    out_name = os.path.join(results_dir, "SinResp_2M.png")
    #plt.savefig(out_name)
    plt.show()

