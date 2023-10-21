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


if __name__ == '__main__':
    # t0 = 1373961618
    t0 = 1374011418  # 21 Jul 2023 21:50:00 UTC

    dur = 7200
    source = '/virgoData/ffl/raw.ffl'

    channels = ['V1:ENV_NEB_SEIS_V', 'V1:Sa_NE_F0_LVDT_V_500Hz', 'V1:Sa_NE_F1_LVDT_V_500Hz', 'V1:Sa_NE_F2_LVDT_V_500Hz',
                'V1:Sa_NE_F3_LVDT_V_500Hz', 'V1:Sa_NE_F4_LVDT_V_500Hz', 'V1:Sa_NE_F7_LVDT_V_500Hz',
                'V1:Sa_NE_F7_LVDT_V1', 'V1:Sa_NE_F7_LVDT_V1', 'V1:Sa_NE_F7_LVDT_V2',
                'V1:Sc_NE_MAR_PSDM_V1', 'V1:Sc_NE_MAR_PSDM_V2',
                'V1:Sc_NE_MAR_PSDT_V1', 'V1:Sc_NE_MAR_PSDT_V2',
                'V1:Sc_NE_MIR_PSDF_V1', 'V1:Sc_NE_MIR_PSDF_V2',
                'V1:Sa_NE_F0_ACC_V1', 'V1:Sa_NE_F0_ACC_V2', 'V1:Sa_NE_F0_ACC_Y_500Hz', 'V1:Sa_NE_F0_Y_500Hz',
                'V1:Sa_NE_F0_Y_CORR_500Hz', 'V1:Sc_NE_MIR_Y_AA', 'V1:Sc_NE_txAA']
    # channels = ['V1:Sa_NE_F0_LVDT_V_500Hz', 'V1:Sa_NE_F7_LVDT_V_500Hz']
    fc = 8.  # Hz
    decim_factor = 8  # decimation factor

    with h5py.File("Safile.hdf5", "w") as fSa:
        lvdt = fSa.create_group('lvdt')
        lv_d = fSa.create_group('lv_d')

        for ch in channels:
            # read frame vector
            vec = framel.frgetvect(source, ch, t0, dur)
            # sampling period
            Ts = vec[3][0]
            fs = 1. / Ts
#           ordbut = signal.buttord(5.e-2, fc, 0.5, 20., analog=False, fs=fs)
            N, Wn = signal.buttord([8.e-2, 8.], [1.e-2, 12.], 3, 40, analog=False, fs=fs)
            b, a = signal.butter(N, Wn, 'band', True)
            # low pass filtering
            sos1 = signal.butter(6, fc, btype='lowpass', analog=False, output='sos', fs=fs)
            # high pass filtering
            # sos1 = signal.butter(6, 5.e-2, btype='highpass', analog=False, output='sos', fs=fs)
            # sos2 = signal.butter(12, Wn, btype='bp', analog=False, output='sos', fs=fs)
            # sos3 = signal.iirdesign([5.e-2, 8.], [2.e-3, 12.], 3, 40, analog=False,
            #                        ftype='ellip', output='sos', fs=fs)
            sos0 = signal.iirdesign(np.float64(10.e-2), np.float64(2.e-3), 1, 60, analog=False,
                                   ftype='ellip', output='sos', fs=fs)
            # signal.freqs(b, a, np.logspace(-3, 2, 500))
            w, h = signal.sosfreqz(sos0, worN=2**16, whole=False, fs=fs)
            plot_filter_tf = False
            if plot_filter_tf:
                plt.figure()
                plt.semilogx(w, 20 * np.log10(abs(h)))
                plt.title('Highpass filter fit to constraints')
                plt.xlabel('Frequency')
                plt.ylabel('Amplitude')
                plt.grid(which='both', axis='both')

            #
            # units
            units = vec[5]
            t_start = vec[1]
            t_series_f0 = signal.sosfilt(sos0, vec[0])
            t_series_f1 = signal.sosfilt(sos1, t_series_f0)
            t_series_dec = signal.decimate(t_series_f1, decim_factor, axis=-1)
            # channel spectrum
            nperseg = 2 ** 18
            channel = ch[6:11]
            plt.figure(channel)
            f, Pxx = signal.welch(vec[0], fs=fs, window='hann', detrend='constant', nperseg=nperseg)
            f_0, P00 = signal.welch(t_series_f0, fs=fs, window='hann', detrend='constant', nperseg=nperseg)
            f_1, P11 = signal.welch(t_series_f1, fs=fs, window='hann', detrend='constant', nperseg=nperseg)
            f_dec, Pyy = signal.welch(t_series_dec, fs=fs / decim_factor, window='hann', detrend='constant', nperseg=nperseg /decim_factor)
            ax = plt.subplot(1, 1, 1)
            ax.loglog(f, np.sqrt(Pxx), 'b', label='raw')
            ax.loglog(f_0, np.sqrt(P00), 'c', label='filt0')
            ax.loglog(f_1, np.sqrt(P11), 'r', label='filt1')
            ax.loglog(f_dec, np.sqrt(Pyy), 'g', label='dec')
            ax.grid(which='both', axis='both')
            ax.set_ylabel(units)
            ax.set_ylim([5.e-4, 5.e0])
            ax.set_xlabel('Frequency Hz')
            ax.legend()

            vl = lvdt.create_dataset(channel, data=vec[0], dtype='float32')
            vl.attrs['t0'] = t0   # initial GPS time
            vl.attrs['channel'] = channel
            vl.attrs['sample_rate'] = fs
            vl.attrs['units'] = units
            print(vl[0])
            print(vl)

            name_dec = channel+'_dec'
            lvd = lv_d.create_dataset(name_dec, data=t_series_dec, dtype='float32')

            lvd.attrs['channel'] = name_dec
            lvd.attrs['sample_rate'] = lvdt[channel].attrs['sample_rate'] / decim_factor
            lvd.attrs['units'] = lvdt[channel].attrs['units']
            print(lvd[0])
            print(lvd)

    plt.show()
