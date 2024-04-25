"""
This code implements the ARMA model for a system composed by 2 masses M1 and M2
and 2 springs K1 and K2. On the system acts a viscous friction force with a coefficient gamma.
In this code is used PID control to control the system variables, so there is a control force
acting on the first mass.

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
    return A @ y + B * u  #nd array of y at instant n+1

#--------------------------Right Hand Side------------------------#
def matrix(gamma, M1, M2, K1, K2, dt):

    #defne the rows of the matrix A
    A_row1 = [1-(dt*gamma/M1), dt*gamma/M1, -dt*(K1+K2)/M1, dt*K2/M1]
    A_row2 = [dt*gamma/M2, 1-(dt*gamma/M2), dt*K2/M2, -dt*K2/M2]
    A_row3 = [dt, 0, 1, 0]
    A_row4 = [0, dt, 0, 1]

    #define A and B
    A = np.array([A_row1, A_row2, A_row3, A_row4])
    B = np.array((dt/M1,0, 0, 0))
    return A, B

#--------------------------Temporal evolution----------------------#
def evolution(evol_method, Nt_step, dt, physical_params, ref, kp, ki, kd, file_name = None):
    #-----------------Initialize the problem-------------------#
    tmax = dt * Nt_step                #total time of simulation
    tt = np.arange(0, tmax, dt)        #temporal grid
    y0 = np.array((0, 0, 1, 0.5))      #initial condition
    y_t = np.copy(y0)                  #create a copy to evolve it in time

    #----------------------Time evolution----------------------#
    v1 = []      #initialize list of v1 value
    v2 = []      # initialize list of v2 value
    x1 = []      #initialize list of x1 value
    x2 = []      # initialize list of x2 value

    #compute the matrices of the system
    A, B = matrix(*physical_params)

    #parameters for PID control
    err_t = []  # list for memorizing the value of the error
    F = 0       # PID term
    I = 0
    j = 0       #cycle index

    #temporal evolution when the ext (control) force is applied
    for t in tt:
        y_t = evol_method(y_t, A, B, F)   #step n+1
        v1.append(y_t[0])
        v2.append(y_t[1])
        x1.append(y_t[2])
        x2.append(y_t[3])

        err = ref - y_t[2]                     # evaluate the error
        err_t.append(err)
        delta_err = err - err_t[j - 1]
        P = kp * err                            # calculate P term
        I = I + ki * (err * dt)                 # calculate the I term
        D = kd * (delta_err / dt)               # calculate the D term
        F = P + I + D                           # calculate PID term
        j = j + 1

    #save simulation's data (if it's necessary)
    if file_name is not None:
        data = np.column_stack((tt, v1, v2, x1, x2))
        np.savetxt(os.path.join(data_dir, file_name), data, header='time, v1, v2, x1, x2')

    return tt, np.array(v1), np.array(v2), np.array(x1), np.array(x2)


if __name__ == '__main__':

    #Parameters of the simulation
    Nt_step = 1e6     #temporal steps
    dt = 1e-3         #temporal step size

    #Parameters of the system
    gamma = 0.5     #viscous friction coeff [kg/m*s]
    M1 = 10        #filter mass [Kg]
    M2 = 10
    K1 = 1        #spring constant [N/m]
    K2 = 1

    #Parameters of control
    ref = 0                                     # reference signal for x1
    control_params_PID = [ref, 150, 0.8, 150]     # ref, kp, ki, kd

    #Simulation
    physical_params = [gamma, M1, M2, K1, K2, dt]
    simulation_params = [AR_model, Nt_step, dt]
    tt, v1_PID, v2_PID, x1_PID, x2_PID = evolution(*simulation_params, physical_params, *control_params_PID, file_name =None)

    # --------------------------Plot results----------------------#
    #fig = plt.figure(figsize=(12,10))
    plt.title('PID control for two coupled oscillators \n x$_1^{ref}$=%.1f, k$_p$=%.1f, k$_i$=%.1f, k$_d$=%.1f'
              %(control_params_PID[0], control_params_PID[1], control_params_PID[2], control_params_PID[3]), size=11)
    plt.xlabel('Time [s]')
    plt.ylabel('position [m]')
    plt.grid(True)
    plt.minorticks_on()

    plt.plot(tt, x1_PID, linestyle='-', linewidth=1, marker='', color='steelblue', label='x1, mass M1')
    plt.plot(tt, x2_PID, linestyle='-', linewidth=1, marker='', color='darkmagenta', label='x2, mass M2')
    plt.legend()
    plt.tight_layout()

    #save the plot in the results dir
    #out_name = os.path.join(results_dir, "SinResp_2M.png")
    #plt.savefig(out_name)
    plt.show()

