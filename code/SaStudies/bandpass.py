import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

#--------------------Setup the main directories------------------#
#Define the various directories
script_dir = os.getcwd()                         #define current dir
main_dir = os.path.dirname(script_dir)           #go up of one directory
results_dir = os.path.join(main_dir, "figure")   #define results dir

if not os.path.exists(results_dir):              #if the directory does not exist create it
    os.mkdir(results_dir)


# Generate a sine signal plus a noise
def signal0(t, A, f, noise):
    return A * np.sin(2 * np.pi * f * t) + noise


if __name__ == '__main__':
    # define time array
    t_start = 0  # start time of the signal [s]
    t_end = 1  # end time [s]
    fs = 800  # sampling frequency of the signal [Hz] --> ts = 1/fs

    t = np.linspace(t_start, t_end, (t_end - t_start) * fs)

    # Parameters of the signal
    f = 4  # sine wave frequency [Hz]
    A = 1  # sine magnitude

    # Generate the noise
    # -------------white noise--------------#
    mean = 0  # center of the distribution
    std = 1  # width of the distribution
    num_samples = len(t)
    white_noise = np.random.normal(mean, std, size=num_samples)

    # generate the signal
    y_gauss = signal0(t, A, f, white_noise)

    #--------------------------------------
    low = 2
    high = 8

    order = 4
    b, a = signal.butter(order, [low, high], btype='band', output='ba', fs=800)
    signal_out = signal.filtfilt(b, a, y_gauss)

    plt.rc('font', size=10)
    plt.figure(figsize=(8, 5))
    plt.title("Filtered signal")
    plt.xlabel("time [s]")
    plt.ylabel("y(t)")
    plt.grid(color='gray', linewidth='0.2')
    plt.minorticks_on()

    plt.plot(t[200:], y_gauss[200:], linestyle='-', linewidth='0.8', marker='', color='gray', label='input signal')
    plt.plot(t[200:], signal_out[200:], linestyle='-', linewidth='1', marker='', color='red', label='filtered signal')
    plt.legend(loc='lower right')
    plt.show()

    #-------------------------------------------