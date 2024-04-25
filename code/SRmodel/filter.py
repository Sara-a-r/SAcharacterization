import os
import numpy as np
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


# Generate a sine signal plus a noise
if __name__ == '__main__':
    # Parameters of the simulation
    Nt_step = 5e5  # temporal steps
    dt = 1e-3  # temporal step size

    # create an array of frequencies
    f = np.linspace(1e-2, 1e1, 10000)
    wm = 2 * np.pi * f
    fs=1000

    # --------------------------------------
    #b, a = signal.iirfilter(N=4, Wn=[0.1, 1.5], btype='bandpass', analog=True, ftype='butter')#, fs=1000)
    #b, a = signal.butter(N=4, Wn=[0.1,1.5], btype='bandpass', analog=False, output='ba', fs=1000)
    #w, h = signal.freqs(b, a, worN=np.linspace(1e-2, 1e1, 10000))#, fs=1000*2*np.pi)

    omega = 2*np.pi*0.106
    zeta = 0.5
    b, a = signal.iirnotch(omega, zeta)
    sys = signal.TransferFunction(b, a)
    # Calcola la risposta in frequenza
    w, h = signal.freqresp(sys, wm)
    #w, h = signal.freqs(b, a, worN=np.linspace(1e-2, 1e1, 10000))  # , fs=1000*2*np.pi)


    # Traccia la risposta in frequenza del filtro
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(w, abs(h))
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude response [dB]')
    plt.grid(True)
    plt.show()

