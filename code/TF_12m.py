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
def TransferFunc (w, M, K, gamma):
    num = np.sqrt(1+gamma**2)
    den = np.sqrt((1-(w**2/(K/M)))**2+gamma**2)
    return num / den


if __name__ == '__main__':
    #create an array of frequencies
    f = np.linspace(0,1e1,100000)
    w = 2*np.pi*f
    #define the parameters of the system
    gamma = 0       # viscous friction coeff [kg/m*s]
    M = 10
    K = 10

    Tf = TransferFunc(w, 0.1, 0.12, 0.01)
    Tf2 = TransferFunc(w, 0.1, 2.0, 0.01)*TransferFunc(w, 0.1, 0.1, 0.01)

    # --------------------------Plot results----------------------#

    plt.title('Transfer function of single and double pendulum', size=13)
    plt.xlabel('Frequency [Hz]', size=12)
    plt.ylabel('TF', size=12)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(0.05,10)
    plt.ylim(10**-6,500)
    plt.grid(True, which='both',ls='-', alpha=0.2, lw=0.5)
    plt.minorticks_on()

    plt.plot(f, Tf, linestyle='-', linewidth=1, marker='', color='steelblue',label='Single pendulum')
    plt.plot(f, Tf2, linestyle='-', linewidth=1, marker='', color= 'coral',label='Double pendulum')
    plt.legend()
    plt.tight_layout()

    #save the plot in the results dir
    #out_name = os.path.join(results_dir, "Tf_2M2K.png")
    #plt.savefig(out_name)
    plt.show()

