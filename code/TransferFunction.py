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
def TransferFunc (w, M1, M2, K1, K2, gamma):
    num = - w**2*M2*K1 + 1j*w*gamma*K1 + K1*K2
    den = w**4*M2*M1 - 1j*w**3*gamma*(M1+M2) - w**2*(M2*(K1+K2)+M1*K2) + 1j*w*gamma*K1 + K1*K2
    return num / den

#----------------------------Phase-----------------------------#
def Phase(Tf):
    return np.arctan(np.imag(Tf)/np.real(Tf)) * 180 /np.pi

#-------------------------Bode plot-----------------------------#
def Bode(Tf):
    return 20 * np.log10(np.abs(Tf))

if __name__ == '__main__':
    #create an array of frequencies
    f = np.linspace(0,1e1,100000)
    w = 2*np.pi*f
    #define the parameters of the system
    gamma = 0.2       # viscous friction coeff [kg/m*s]
    M1 = 20         # filter mass [Kg]
    M2 = 20
    K1 = 10          # spring constant [N/m]
    K2 = 10

    Tf = TransferFunc(w,M1, M2, K1, K2, gamma)
    M = (np.real(Tf)**2+np.imag(Tf)**2)**(1/2)
    A = Bode(Tf)

    # --------------------------Plot results----------------------#

    plt.title('Transfer function for coupled oscillators \n M$_1$=%d, M$_2$=%d, K$_1$=%d, K$_2$=%d, $\gamma$=%.1f' % (M1,M2,K1,K2,gamma))
    plt.xlabel('f [Hz]')
    plt.ylabel('|x$_1$/x$_0$|')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True)
    plt.minorticks_on()

    plt.plot(f, M, linestyle='-', linewidth=1, marker='', color='steelblue')
    plt.tight_layout()

    #save the plot in the results dir
    out_name = os.path.join(results_dir, "Tf_2M2K.png")
    plt.savefig(out_name)
    plt.show()

