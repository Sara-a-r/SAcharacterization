import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import linalg

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

#-------------------PCA function----------------#
def cov(X):
    return np.dot(X.T, X) / X.shape[0]

def pca(data, pc_count=None):
    data -= np.mean(data, 0)
    data /= np.std(data, 0)
    C = cov(data)
    E, V = linalg.eigh(C)
    key = np.argsort(E)[::-1][:pc_count]
    E, V = E[key], V[:, key]
    U = np.dot(data, V)  # used to be dot(V.T, data.T).T
    return U, E, V

#------------------------------------------------#

if __name__ == '__main__':

    # upload data
    data_t0 = np.loadtxt(os.path.join(data_dir, "LVDT_t0"), unpack=True)
    data_t1 = np.loadtxt(os.path.join(data_dir, "LVDT_t1"), unpack=True)
    data_t2 = np.loadtxt(os.path.join(data_dir, "LVDT_t2"), unpack=True)
    data_t3 = np.loadtxt(os.path.join(data_dir, "LVDT_t3"), unpack=True)
    data_t4 = np.loadtxt(os.path.join(data_dir, "LVDT_t4"), unpack=True)

    #--------digital low pass filter---------#
    low, high = 0.09, 0.2
    order = 4
    b, a = signal.butter(order, Wn = high, btype='lowpass', output='ba', fs=62.5)
    data_filtered0 = signal.filtfilt(b, a, data_t0)
    data_filtered1 = signal.filtfilt(b, a, data_t1)
    data_filtered2 = signal.filtfilt(b, a, data_t2)
    data_filtered3 = signal.filtfilt(b, a, data_t3)
    data_filtered4 = signal.filtfilt(b, a, data_t4)

    f, Pxx0 = signal.welch(data_filtered0, fs=62.5, window='hann', nperseg=2**16)
    _, Pxx1 = signal.welch(data_filtered1, fs=62.5, window='hann', nperseg=2 ** 16)
    _, Pxx2 = signal.welch(data_filtered2, fs=62.5, window='hann', nperseg=2 ** 16)
    _, Pxx3 = signal.welch(data_filtered3, fs=62.5, window='hann', nperseg=2 ** 16)
    _, Pxx4 = signal.welch(data_filtered4, fs=62.5, window='hann', nperseg=2 ** 16)

    psd_data0 = np.sqrt(Pxx0)
    psd_data1 = np.sqrt(Pxx1)
    psd_data2 = np.sqrt(Pxx2)
    psd_data3 = np.sqrt(Pxx3)
    psd_data4 = np.sqrt(Pxx4)

    #------------pca analysis-----------------#
    list_psd = [psd_data0, psd_data1, psd_data2, psd_data3, psd_data4]
    list_time = [data_filtered0, data_filtered1, data_filtered2, data_filtered3, data_filtered4]
    vpos = np.array([i for i in list_time]).T
    #print(vpos)
    U, E, V = pca(vpos, 6)
    with np.printoptions(precision=3, linewidth=160):
        print('U')
        print(U)
        print('E')
        print(E)
        print('V')
        print(V)

    print('Le frequenze sono :\n')
    for iU in range(len(E)):
        f, Pxx = signal.welch(U[:, iU], fs=62.5, window='hann', nperseg=2**16)
        psd_U = np.sqrt(Pxx)
        indici = np.where((psd_U>=3.8) & (psd_U<=4.2))
        print(f[indici])



    #------------------------------------------#

    plt.xlabel('f [Hz]')
    plt.ylabel('')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True)
    plt.minorticks_on()

    #plt.plot(ff_m, data_m)
    #plt.plot(f, psd_U)
    #plt.plot(f, psd_data)
    #plt.plot(data_filtered)
    #plt.plot(data_t)

    #plt.show()

    # Definizione delle costanti M1, M2, M3, M4, M5
    M = [173, 165, 140, 118, 315]
    w_values = [0.69, 1.01, 2.20, 3.33, 5.47]