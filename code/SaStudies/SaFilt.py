# import numerical analysis, graphics and statistics modules
import numpy as np
# import scipy
# from scipy import io, integrate, linalg
from scipy import signal

# from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
# import matplotlib.colors as colors
# import scipy.optimize as opt
# from scipy.stats import norm
# from numpy.random import default_rng
# import csv as csv
# import sympy as sym
# from sympy.solvers import solve
# from sympy import factor_terms

import framel

plt.rc('font', size=12)
# Clean up of old figures
plt.close('all')

def correl(ch1,ch2,nperseg):
    f, Pxx = signal.welch(ch1, fs=500., window='hann', nperseg=nperseg)
    f, Pyy = signal.welch(ch2, fs=500., window='hann', nperseg=nperseg)
    f, Pxy = signal.csd(ch1, ch2, fs=500., window='hann', nperseg=nperseg)
    f, Cxy = signal.coherence(ch1, ch2, fs=500., window='hann', nperseg=nperseg)
    # spectrum channel 1
    plt.figure()
    ax = plt.subplot(2,1,1)
    ax.loglog(f, np.sqrt(Pxx))
    ax.grid()
    ax.set_ylabel(ch1n)
    ax.set_xlabel('Frequency Hz')
    # spectrum channel 2
    ax1 = plt.subplot(2,1,2)
    ax1.loglog(f, np.sqrt(Pyy))
    ax1.grid()
    ax1.set_ylabel(ch2n)
    ax1.set_xlabel('Frequency Hz')
    #
    plt.figure()
    # coherence
    ax20 = plt.subplot(3,1,1)
    ax20.semilogx(f, np.sqrt(Cxy))
    ax20.set_ylim([0.,1.])
    ax20.grid()
    ax20.set_ylabel('C '+ch1n+' vs '+ch2n)
    ax20.set_xlabel('Frequency Hz')
    # amplitude
    ax21 = plt.subplot(3,1,2)
    ax21.loglog(f, np.sqrt(np.abs(Pxy)))
    ax21.grid()
    ax21.set_ylabel('TF ')
    ax21.set_xlabel('Frequency Hz')
    # phase
    ax22 = plt.subplot(3,1,3)
    ax22.semilogx(f, np.angle(Pxy)*180/np.pi)
    ax22.grid()
    ax22.set_ylim([-180.,180.])
    ax22.set_ylabel('Phase ')
    ax22.set_xlabel('Frequency Hz')

if __name__ == '__main__':
    # Sa_NE_F0_ACC_500Hz = framel.frgetvect('/virgoData/ffl/raw.ffl','V1:Sa_NE_F0_ACC_500Hz',1373383300,100)
    # t0 = 1373383300
    t0 = 1373961618
    dur = 1000
    NE_F0 = framel.frgetvect('/virgoData/ffl/raw.ffl', 'V1:Sa_NE_F0_LVDT_V_500Hz', t0, dur)
    # NE_F1 = framel.frgetvect('/virgoData/ffl/raw.ffl', 'V1:Sa_NE_F1_LVDT_V_500Hz', t0, dur)
    # NE_F2 = framel.frgetvect('/virgoData/ffl/raw.ffl', 'V1:Sa_NE_F2_LVDT_V_500Hz', t0, dur)
    # NE_F3 = framel.frgetvect('/virgoData/ffl/raw.ffl', 'V1:Sa_NE_F3_LVDT_V_500Hz', t0, dur)
    # NE_F4 = framel.frgetvect('/virgoData/ffl/raw.ffl', 'V1:Sa_NE_F4_LVDT_V_500Hz', t0, dur)
    # NE_F7 = framel.frgetvect('/virgoData/ffl/raw.ffl', 'V1:Sa_NE_F7_LVDT_V_500Hz', t0, dur)
    # NE_F0_ACC1 = framel.frgetvect('/virgoData/ffl/raw.ffl', 'V1:Sa_NE_F0_ACC_V1_500Hz', t0, dur)
    # NE_F0_ACC2 = framel.frgetvect('/virgoData/ffl/raw.ffl', 'V1:Sa_NE_F0_ACC_V2_500Hz', t0, dur)
    # low pass filtering
    # sampling frequency
    fs = 1. / NE_F0[3][0]
    sos = signal.butter(6, 10., btype='low', analog=False, output='sos', fs=fs)
    time = NE_F0[1] + NE_F0[3] * np.array(range(0, len(NE_F0[0])))
    ch1 = np.array(NE_F0[0])
    ch1filt = signal.sosfilt(sos, ch1)
    #  ch1_2 = signal.decimate(ch1, 2, axis=-1)
    #  ch1_4 = signal.decimate(ch1_2, 2, axis=-1)
    # ch1_8 = signal.decimate(ch1_4, 2, axis=-1)
    # ch1_4 = signal.decimate(ch1_2, 2, axis=-1)
    ch1_8 = signal.decimate(ch1filt, 8, axis=-1)
    ch1n = 'NE_F0'
    nperseg = 2 ** 14

    plt.figure()
    f, Pxx = signal.welch(ch1, fs=500., window='hann', nperseg=nperseg)
    f, Pyy = signal.welch(ch1filt[50:], fs=500., window='hann', nperseg=nperseg)
    f10, Py10 = signal.welch(ch1_8[5:], fs=500./8., window='hann', nperseg=nperseg)
    # spectrum channel 1
    ax = plt.subplot(1, 1, 1)
    ax.loglog(f, np.sqrt(Pxx))
    ax.loglog(f, np.sqrt(Pyy))
    ax.loglog(f10, np.sqrt(Py10))
    ax.grid()
    ax.set_ylabel(ch1n)
    ax.set_ylim([5.e-4, 5.e0])
    ax.set_xlabel('Frequency Hz')
    #
    plt.show()
