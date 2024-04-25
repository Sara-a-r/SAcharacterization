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

import h5py
# from gwdama.io import GwDataManager
import framel


def psd(chi, chin, nperseg):
    f, Pxx = signal.welch(chi, fs=500., window='hann', nperseg=nperseg)

    # spectrum
    ax = plt.subplot(3,2,iplot+1)
    # cumulated RMS
    df = np.diff(f)
    Varxx = np.cumsum(np.flip(df * Pxx[0:-1]))
    ax.loglog(f, np.sqrt(Pxx))
    ax.loglog(f[0:-1], np.flip(np.sqrt(Varxx)))
    ax.grid()
    ax.set_ylabel(chin)
    # ax.set_xlabel('Frequency Hz')

if __name__ == '__main__':
    # Sa_NE_F0_ACC_500Hz = framel.frgetvect('/virgoData/ffl/raw.ffl','V1:Sa_NE_F0_ACC_500Hz',1373383300,100)
    # t0 = 1373383300
    t0 = 1373961618
    dur = 1000
    NE_F0 = framel.frgetvect('/virgoData/ffl/raw.ffl', 'V1:Sa_NE_F0_LVDT_V_500Hz', t0, dur)
    #  NE_F1 = framel.frgetvect('/virgoData/ffl/raw.ffl', 'V1:Sa_NE_F1_LVDT_V_500Hz', t0, dur)
    #  NE_F2 = framel.frgetvect('/virgoData/ffl/raw.ffl', 'V1:Sa_NE_F2_LVDT_V_500Hz', t0, dur)
    #  NE_F3 = framel.frgetvect('/virgoData/ffl/raw.ffl', 'V1:Sa_NE_F3_LVDT_V_500Hz', t0, dur)
    #  NE_F4 = framel.frgetvect('/virgoData/ffl/raw.ffl', 'V1:Sa_NE_F4_LVDT_V_500Hz', t0, dur)
    #  NE_F7 = framel.frgetvect('/virgoData/ffl/raw.ffl', 'V1:Sa_NE_F7_LVDT_V_500Hz', t0, dur)
    # NE_F0_ACC1 = framel.frgetvect('/virgoData/ffl/raw.ffl', 'V1:Sa_NE_F0_ACC_V1_500Hz', t0, dur)
    # NE_F0_ACC2 = framel.frgetvect('/virgoData/ffl/raw.ffl', 'V1:Sa_NE_F0_ACC_V2_500Hz', t0, dur)
    # low pass filtering
    # sampling frequency
    a0 = NE_F0[0]
    a3 = NE_F0[3][0]
    fs = 1. / NE_F0[3][0]
    sos = signal.butter(6, 4., btype='low', analog=False, output='sos', fs=fs)
    time = NE_F0[1] + NE_F0[3] * np.array(range(0, len(NE_F0[0])))
    ch1 = NE_F0[0]
    ch1filt = signal.sosfilt(sos, ch1)
    ch1n = 'NE_F0'
    nperseg = 2 ** 14

    plt.figure()
    f, Pxx = signal.welch(ch1, fs=500., window='hann', nperseg=nperseg)
    f, Pyy = signal.welch(ch1filt, fs=500., window='hann', nperseg=nperseg)
    # spectrum channel 1
    ax = plt.subplot(1, 1, 1)
    ax.loglog(f, np.sqrt(Pxx))
    ax.loglog(f, np.sqrt(Pyy))
    ax.grid()
    ax.set_ylabel(ch1n)
    ax.set_ylim([1.e-8, 1.e-5])
    ax.set_xlabel('Frequency Hz')

    ch = []
    chn = []
    ch.append(NE_F0[0])
    chn.append('NE_F0')
    ch.append(NE_F1[0])
    chn.append('NE_F1')
    ch.append(NE_F2[0])
    chn.append('NE_F2')
    ch.append(NE_F3[0])
    chn.append('NE_F3')
    ch.append(NE_F4[0])
    chn.append('NE_F4')
    ch.append(NE_F7[0])
    chn.append('NE_F7')
    nperseg = 2 ** 14

    plt.figure('Spectra')

    #  with h5py.File("mytestfile.hdf5", "w") as f:
    #      dset = f.create_dataset("mydataset", (100,), dtype='i')

    for chi, chni, iplot in zip(ch, chn, range( len(ch) )):
        psd(chi, chni, nperseg)

        gps_start = '2023-7-8 8:00'
        gps_end = '2023-7-8 8:02'
        # List of channels
        channels = ['V1:Sa_NE_F0_LVDT_V_500Hz', 'V1:Sa_NE_F1_LVDT_V_500Hz', 'V1:Sa_NE_F2_LVDT_V_500Hz',
                    'V1:Sa_NE_F3_LVDT_V_500Hz', 'V1:Sa_NE_F4_LVDT_V_500Hz', 'V1:Sa_NE_F7_LVDT_V_500Hz']

        with h5py.File("Safile.hdf5", "a") as fSa:
            vlvdt = fSa.create_dataset('vlvdt', data=chi, dtype='float64')

            vlvdt.attrs['t0'] = t0  #  initial GPS time
            vlvdt.attrs['ch'] = chn[0]
            vlvdt.attrs['channel'] = channels[0]
            vlvdt.attrs['sample_rate'] = 1. / ch[0][3]
            vlvdt.attrs['unit'] = ch[0][5]
            print(vlvdt[0])

            print(vlvdt)


    # dama = GwDataManager('SaData')
    # dama.create_dataset(chn[0], data=ch[0][0])
    # dama['t0'] =
    # dama.read_gwdata(gps_start, gps_end, channels=channels,
    #                      data_source='local',
    #                      dts_key='Raw',                                  # a key for the group of channels
    #                      nproc=10, verbose=True,                         # To speed it up and check what's happening
    #                      resample=125)
    plt.show()