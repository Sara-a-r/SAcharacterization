"""
This code implements the FFT and the PSD using python libraries (scipy).
Data are taken from the simulation.
"""


import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy import signal as sgnl

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

#------------------------------FFT-----------------------------------#
def FFT(fs, signal):
    freqs = fftpack.fftfreq(len(signal)) * fs
    Y = fftpack.fft(signal)

    freqs = freqs[np.where(freqs > 0)]      #consider only positive freq
    Y = np.abs(Y)[np.where(freqs > 0)]
    return freqs, Y

if __name__ == '__main__':

    #upload data
    tt, v1, v2, v3, v4, v5, x1, x2, x3, x4, x5 = np.loadtxt(os.path.join(data_dir, "5M5Ksimulation.txt"), unpack=True)

    #-----------------------------Compute FFT----------------------------#
    #compute the FFT
    fs = 1000                    #sampling frequency (fs>2fmax)
    freqs, Y1 = FFT(fs, x1)
    _, Y2 = FFT(fs, x2)
    _, Y3 = FFT(fs, x3)
    _, Y4 = FFT(fs, x4)
    _, Y5 = FFT(fs, x5)


    #Plot results
    plt.rc('font', size=10)
    plt.figure(figsize=(8, 5))
    plt.title("FFT analysis")
    plt.xlabel("f [Hz]")
    plt.ylabel("Frequency Domain (Spectrum) Magnitude")
    #plt.xlim(0,fs/2)
    plt.grid(color='gray', linewidth='0.2')
    plt.minorticks_on()

    plt.plot(freqs, Y1, linestyle='-', linewidth='0.8', marker='', color='steelblue', label='mass M1')
    plt.plot(freqs, Y5, linestyle='-', linewidth='0.8', marker='', color='darkmagenta', label='mass M5')
    plt.legend()
    #plt.tight_layout()

    #-------------------------------PSD--------------------------------#
    # compute the power spectral density using Welch’s method
    freq, PSD_x1 = sgnl.welch(x1, fs, nperseg=2**14)
    _, PSD_x2 = sgnl.welch(x2, fs, nperseg=2 ** 14)
    _, PSD_x3 = sgnl.welch(x3, fs, nperseg=2 ** 14)
    _, PSD_x4 = sgnl.welch(x4, fs, nperseg=2 ** 14)
    _, PSD_x5 = sgnl.welch(x5, fs, nperseg=2 ** 14)

    plt.rc('font', size=10)
    plt.figure(figsize=(8, 5))
    plt.title("PSD of the two masses system")
    plt.xlabel("f [Hz]")
    plt.ylabel("PSD [1/Hz]")
    plt.grid(color='gray', linewidth='0.2')
    plt.minorticks_on()

    plt.semilogy(freq, PSD_x1, linestyle = '-', linewidth = '0.8', marker = '', color = 'steelblue', label = 'M1')
    plt.semilogy(freq, PSD_x5, linestyle='-', linewidth='0.8', marker='', color='darkmagenta', label='M5')
    plt.legend()

    #save the plot in the results dir
    out_name = os.path.join(results_dir, "FreqAna_5M.png")
    #plt.savefig(out_name)
    plt.show()

