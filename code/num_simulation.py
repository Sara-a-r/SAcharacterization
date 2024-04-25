"""
This file uses Euler method to solve the system of equation:
    d_t ( w ) = -b/J ( w ) + Kt/J ( i )
    d_t ( i ) = -Ke/L ( w ) - R/L ( i ) + V/L
    d_t ( theta ) = ( w )
that can be written as follows
    d_t ( u ) = A ( u ) + B
where A = [-b/J   Kt/J  0]  and  B = [0  ]
          [-Ke/L  -R/L  0]           [V/L]
          [1        0   0]           [0  ]

Note: the inizial condition are w=0, i=0, theta=0
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

#-----------------Integrator: Euler method------------------------#
def euler_step(u, F, dt, *params):
    return u + dt * F(u, *params)  #nd array of u at instant n+1

#--------------------------Right Hand Side------------------------#
def F(u, K1, K2, M1, M2):
    #define A and B

    # defne the rows of the matrix A
    A_row1 = [0, 0, - (K1 + K2) / M1, K2 / M1]
    A_row2 = [0, 0, K2 / M2, -K2 / M2]
    A_row3 = [1, 0, 0, 0]
    A_row4 = [0, 1, 0, 0]

    # define A and B
    A = np.array([A_row1, A_row2, A_row3, A_row4])
    B = np.array((0, 0, 0, 0))
    return A @ u + B

#----------------------------Step function------------------------#
#def step_function(t, t0=0):
 #   return np.heaviside(t-t0, 1)    # 0 if t < t0
                                    # 1 if t >= t0

#--------------------------Temporal evolution----------------------#
def evolution(int_method, Nt_step, dt, physics_params):
    #-----------------Initialize the problem-------------------#
    tmax = dt * Nt_step             #total time of simulation
    tt = np.arange(0, tmax, dt)     #temporal grid
    u0 = np.array((0, 0, 1, 1.62))           #initial condition
    u_t = np.copy(u0)               #create a copy to evolve it in time
   # V_signal = V(tt, t0)            #signal applied to the system in time
    #----------------------------------------------------------#

    #----------------------Time evolution----------------------#
    v1 = []  # initialize list of v1 value
    v2 = []  # initialize list of v2 value
    x1 = []  # initialize list of x1 value
    x2 = []  # initialize list of x2 value
    for t in tt:
        u_t = int_method(u_t, F, dt, *physics_params) #step n+1
        v1.append(u_t[0])
        v2.append(u_t[1])
        x1.append(u_t[2])
        x2.append(u_t[3])
    return tt, np.array(v1), np.array(v2), np.array(x1), np.array(x2)


if __name__ == '__main__':
    #Parameters of the simulation
    Nt_step = 2e5     #temporal steps
    dt = 1e-3         #temporal step size
    # Parameters of the system
    gamma = 0  # viscous friction coeff [kg/m*s]
    M1 = 10  # filter mass [Kg]
    M2 = 10
    K1 = 100  # spring constant [N/m]
    K2 = 100
    t0 = 0  # parameter of the step function [s]
    #F0 = 0  # amplitude of the external force
    #w = 10  # f of the ext force

    #Signal applied to the system
    #V = step_function

    #Simulation
    physical_params = [K1, K2, M1, M2]
    simulation_params = [euler_step, Nt_step, dt]
    tt, v1, v2, x1, x2 = evolution(*simulation_params, physical_params)

    #--------------------------Plot results----------------------#
    plt.title('Step response for DC motor (open loop)')
    plt.xlabel('Time [s]')
    plt.ylabel('$\Theta$ [rad]')
    plt.grid(True)
    plt.minorticks_on()
    plt.plot(tt, x1, linestyle='-', linewidth=1.8, marker='')

    # save the plot in the results dir
    out_name = os.path.join(results_dir, "StepResp_theta_numSim1.png")
    #plt.savefig(out_name)
    plt.show()

