"""
This code implements the ARMA model for a system composed by 5 masses and 2 springs K1 and K2.
On the system acts a viscous friction force with a coefficient gamma.
In this code PID control is used to control the system variables, so there is a control force
acting on the first mass and another control force on the third mass.

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
def matrix(M1, M2, M3, M4, M5, K1, K2, K3, K4, K5, g2, g3, g4, g5, dt):
    # defne the matrices A and B
    Id = np.eye(5)
    V = np.array([[1-(dt*g2/M1), dt*g2/M1, 0, 0, 0],
                       [dt*g2/M2, 1-dt*(g2+g3)/M2, dt*g3/M2, 0, 0],
                       [0, dt*g3/M3, 1-dt*(g3+g4)/M3, dt*g4/M3, 0],
                       [0, 0, dt*g4/M4, 1-dt*(g4+g5)/M4, dt*g5/M4],
                       [0, 0, 0, dt*g5/M5, 1-dt*g5/M5]])
    X = dt * np.array([[-(K1+K2)/M1, K2/M1, 0, 0, 0],
                       [K1/M2, -(K2+K3)/M2, K3/M2, 0, 0],
                       [0, K3/M3, -(K3+K4)/M3, K4/M3, 0],
                       [0, 0, K4/M4, -(K4+K5)/M4, K5/M4],
                       [0, 0, 0, K5/M5, -K5/M5]])
    A = np.block([[V, X],
                  [dt*Id, Id]])

    B = np.array((dt/M1,0, dt/M3, 0, dt/M5, 0, 0, 0, 0, 0))
    return A, B

#--------------------------Temporal evolution----------------------#
def evolution(evol_method, Nt_step, dt, physical_params, ref, kp, ki, kd, file_name = None):
    #-----------------Initialize the problem-------------------#
    tmax = dt * Nt_step                                 #total time of simulation
    tt = np.arange(0, tmax, dt)                         #temporal grid
    y0 = np.array((0, 0, 0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5))       #initial condition
    y_t = np.copy(y0)                                   #create a copy to evolve it in time

    #----------------------Time evolution----------------------#
    v1, v2, v3, v4, v5 = [[], [], [], [], []]  # initialize list of v values
    x1, x2, x3, x4, x5 = [[], [], [], [], []]  # initialize list of x values

    #compute the matrices of the system
    A, B = matrix(*physical_params)

    #parameters for PID control
    err_t1, err_t3, err_t5 = [], [], []   # list for memorizing the value of the error
    I1, I3, I5 = 0, 0, 0
    j = 0                      # cycle index

    # initialize u vector
    u = np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0))

    #temporal evolution when the ext (control) force is applied
    for t in tt:
        y_t = evol_method(y_t, A, B, u)   #step n+1
        v1.append(y_t[0])
        v2.append(y_t[1])
        v3.append(y_t[2])
        v4.append(y_t[3])
        v5.append(y_t[4])
        x1.append(y_t[5])
        x2.append(y_t[6])
        x3.append(y_t[7])
        x4.append(y_t[8])
        x5.append(y_t[9])

        #PID control for the first mass
        err1 = ref - y_t[5]  # evaluate the error
        err_t1.append(err1)
        delta_err1 = err1 - err_t1[j - 1]
        P1 = kp[0] * err1  # calculate P term
        I1 = I1 + ki[0] * (err1 * dt)  # calculate the I term
        D1 = kd[0] * (delta_err1 / dt)  # calculate the D term
        Fc1 = P1 + I1 + D1  # calculate PID term

        # PID control for the third mass
        err3 = ref - y_t[7]  # evaluate the error
        err_t3.append(err3)
        delta_err3 = err3 - err_t3[j - 1]
        P3 = kp[1] * err3  # calculate P term
        I3 = I3 + ki[1] * (err3 * dt)  # calculate the I term
        D3 = kd[1] * (delta_err3 / dt)  # calculate the D term
        Fc3 = P3 + I3 + D3  # calculate PID term

        # PID control for the third mass
        err5 = ref - y_t[9]  # evaluate the error
        err_t5.append(err5)
        delta_err5 = err5 - err_t5[j - 1]
        P5 = kp[2] * err5  # calculate P term
        I5 = I5 + ki[2] * (err5 * dt)  # calculate the I term
        D5 = kd[2] * (delta_err5 / dt)  # calculate the D term
        Fc5 = P5 + I5 + D5  # calculate PID term

        u = np.array((Fc1, 0, Fc3, 0, Fc5, 0, 0, 0, 0, 0))

        j = j + 1

    #save simulation's data (if it's necessary)
    if file_name is not None:
        data = np.column_stack((tt, v1, v2, v3, v4, v5, x1, x2, x3, x4, x5))
        np.savetxt(os.path.join(data_dir, file_name), data, header='time, v1, v2, v3, v4, v5, x1, x2, x3, x4, x5')

    return tt, np.array(v1), np.array(v2), np.array(v3), np.array(v4), np.array(v5), np.array(x1), np.array(x2), np.array(x3), np.array(x4), np.array(x5)


if __name__ == '__main__':

    #Parameters of the simulation
    Nt_step = 1e6     #temporal steps
    dt = 1e-3         #temporal step size

    #Parameters of the system
    gamma = [0.5, 0.5, 0.5, 0.5]  # viscous friction coeff [kg/m*s]
    M = [10, 10, 10, 10, 10]  # filter mass [Kg]  [M1, M2, M3, M4, M5]
    K = [1, 1, 1, 1, 1]  # spring constant [N/m]  [K1, K2, K3, K4, K5]

    #Parameters of control
    ref = 0                                                     # reference signal for x1 and x3
    control_params_PID = [ref, (30, 30, 50), (10, 10, 20), (30, 30, 50)]    # ref, kp, ki, kd

    #Simulation
    physical_params = [*M, *K, *gamma, dt]
    simulation_params = [AR_model, Nt_step, dt]
    tt, v1_PID, v2_PID, v3_PID, v4_PID, v5_PID, x1_PID, x2_PID, x3_PID, x4_PID, x5_PID = evolution(*simulation_params,
                                                           physical_params, *control_params_PID,
                                                           file_name=None)

    # --------------------------Plot results----------------------#
    #fig = plt.figure(figsize=(12,10))
    plt.title('PID control for five coupled oscillators (M$_1$, M$_3$, M$_5$) \n x$_1^{ref}$=%.1f, k$_p$=(%.1f,%.1f,%.1f), k$_i$=(%.1f,%.1f,%.1f), k$_d$=(%.1f,%.1f,%.1f)'
              %(control_params_PID[0], control_params_PID[1][0], control_params_PID[1][1], control_params_PID[1][2],
                control_params_PID[2][0], control_params_PID[2][1], control_params_PID[2][2],control_params_PID[3][0],
                control_params_PID[3][1],control_params_PID[3][2]), size=11)
    #plt.title('Time evolution for five coupled oscillators')
    plt.xlabel('Time [s]')
    plt.ylabel('position [m]')
    plt.grid(True)
    plt.minorticks_on()

    plt.plot(tt, x1_PID, linestyle='-', linewidth=1, marker='', color='steelblue', label='x1, mass M$_1$')
    plt.plot(tt, x2_PID, linestyle='-', linewidth=1, marker='', color='black', label='x2, mass M$_2$')
    plt.plot(tt, x3_PID, linestyle='-', linewidth=1, marker='', color='red', label='x3, mass M$_3$')
    plt.plot(tt, x4_PID, linestyle='-', linewidth=1, marker='', color='green', label='x4, mass M$_4$')
    plt.plot(tt, x5_PID, linestyle='-', linewidth=1, marker='', color='darkmagenta', label='x5, mass M$_5$')
    plt.legend()
    plt.tight_layout()

    #save the plot in the results dir
    #out_name = os.path.join(results_dir, "SinResp_2M.png")
    #plt.savefig(out_name)
    plt.show()

