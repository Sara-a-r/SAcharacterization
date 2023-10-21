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


def plot_filter_tf():
    """Plot LP and HP filter TF"""

    plt.figure('Transfer functions')
    w, h = signal.sosfreqz(sos0, worN=2 ** 16, whole=False, fs=fs)
    # plot amplitude
    axtf = plt.subplot(2, 2, 1)
    axtf.semilogx(w, 20 * np.log10(abs(h)))
    axtf.set_title('High pass filter fit to constraints')
    axtf.set_xlabel('Frequency')
    axtf.set_ylabel('Amplitude')
    axtf.grid(which='both', axis='both')
    # plot phase
    axtf = plt.subplot(2, 2, 2)
    axtf.semilogx(w, np.angle(h))
    axtf.set_title('High pass filter fit to constraints')
    axtf.set_xlabel('Frequency')
    axtf.set_ylabel('Phase')
    axtf.grid(which='both', axis='both')

    w, h = signal.sosfreqz(sos1, worN=2 ** 16, whole=False, fs=fs)
    # plot amplitude
    axtf = plt.subplot(2, 2, 3)
    axtf.semilogx(w, 20 * np.log10(abs(h)))
    axtf.set_title('Low pass filter fit to constraints')
    axtf.set_xlabel('Frequency')
    axtf.set_ylabel('Amplitude')
    axtf.grid(which='both', axis='both')
    # plot phase
    axtf = plt.subplot(2, 2, 4)
    axtf.semilogx(w, np.angle(h))
    axtf.set_title('Low pass filter fit to constraints')
    axtf.set_xlabel('Frequency')
    axtf.set_ylabel('Phase')
    axtf.grid(which='both', axis='both')

if __name__ == '__main__':
    # t0 = 1373961618
    # t0 = 1374011418  # 21 Jul 2023 21:50:00 UTC
    t0 = 1377244818  # 28 Aug 2023  8:00:00 UTC

    dur = 720
    source = '/virgoData/ffl/raw.ffl'

    # towers
    towers = ['BS', 'NI', 'WI', 'PR', 'SR', 'NE', 'WE']
    # buildings
    buildings = {'BS': 'CEB', 'PR': 'CEB', 'SR': 'CEB', 'NI': 'CEB', 'WI': 'CEB', 'NE': 'NEB', 'WE': 'CEB'}
    # building environmental channels
    env = {'CEB': 'V1:ENV_CEB_SEIS_V', 'NEB': 'V1:ENV_NEB_SEIS_V', 'WEB': 'V1:ENV_WEB_SEIS_V'}
    # seismic filters
    sfilters = ['F0', 'F1', 'F2', 'F3', 'F4', 'F7']
    # vertical distance
    lvdt = {'F0': '_LVDT_V_500Hz',
            'F1': '_LVDT_V_500Hz',
            'F2': '_LVDT_V_500Hz',
            'F3': '_LVDT_V_500Hz',
            'F4': '_LVDT_V_500Hz',
            'F7': '_LVDT_V_500Hz'}

    # accelerometers
    f0accel = ['CRBAR1', 'CRBAR2', 'CRBAR']
    accel = {'CRBAR1': '_F0_ACC_V1_500Hz', 'CRBAR2': '_F0_ACC_V2_500Hz', 'CRBAR': '_F0_ACC_Y_500Hz'}
    possd = {'MAPSDM1': '_MAR_PSDM_Y1', 'MAPSDM2': '_MAR_PSDM_Y2',
             'MAPSDT1': '_MAR_PSDT_Y1', 'MAPSDT2': '_MAR_PSDT_Y2',
             'MIPSDF1': '_MIR_PSDF_Y1', 'MIPSDF2': '_MIR_PSDF_Y2'}
    others = {'F7LV1': '_F7_LVDT_V1', 'F7LV2': '_F7_LVDT_V2', 'F7LV3': '_F7_LVDT_V3'}
    corrsa = {'F0': 'F0_Y_CORR_500Hz'}
    corrsc = {'MIR': 'MIR_Y_AA', 'txAA': 'txAA'}

    # original sampling frequency
    fs = 500.  # Hz
    # high frequency sampling
    fs10k = 1.e4  # Hz
    # medium frequency sampling
    fs1k = 1.e3  # Hz
    # target sampling frequency
    fs_targ = 62.5  # Hz

    # filters set up
    # high pass filtering
    sos010k = signal.iirdesign(np.float64(10.e-2), np.float64(2.e-3), 1, 60, analog=False,
                            ftype='ellip', output='sos', fs=fs10k)
    sos01k = signal.iirdesign(np.float64(10.e-2), np.float64(2.e-3), 1, 60, analog=False,
                            ftype='ellip', output='sos', fs=fs1k)
    sos0 = signal.iirdesign(np.float64(10.e-2), np.float64(2.e-3), 1, 60, analog=False,
                            ftype='ellip', output='sos', fs=fs)
    # low pass filtering from 500 Hz to 62.5 Hz: cut above 8 Hz and decimate
    fc = 8.  # Hz
    sos1 = signal.butter(6, fc, btype='lowpass', analog=False, output='sos', fs=fs)
    decim_factor = 8

    for tt in ['NE']:  #towers:
        chlist = []
        # environmental channels
        chlist.append(env[buildings[tt]])
        # filter lvdt channels
        for sf in sfilters:
            chlist.append('V1:Sa_' + tt + '_' + sf + lvdt[sf])
        # accelerometers
        for ch in others:
            chlist.append('V1:Sc_' + tt + others[ch])
        # position sensitive devices
        for psd in possd:
            chlist.append('V1:Sc_' + tt + possd[psd])
        for acc in accel:
            chlist.append('V1:Sa_' + tt + accel[acc])
        for corr in corrsa:
            chlist.append('V1:Sa_' + tt + '_' + corrsa[corr])
        for corr in corrsc:
            chlist.append('V1:Sc_' + tt + '_' + corrsc[corr])
        [print(ch) for ch in chlist]

        with h5py.File("Sa" + tt + "_test.hdf5", "w") as fSa:
            sa = fSa.create_group(tt)
            env5 = sa.create_group('env')
            sf5 = sa.create_group('sf')
            sflvdt5 = sf5.create_group('lvdt')
            others5 = sa.create_group('others')
            # position sensitive devices
            psd5 = sa.create_group('psd')
            # accelerometers
            acc5 = sa.create_group('acc')
            # correction signals
            corrsa5 = sa.create_group('corr_sa')
            corrsc5 = sa.create_group('corr_sc')

            for ch in chlist:  # ['V1:Sc_NE_MIR_PSDF_Y2' ]
                # read frame vector
                vec = framel.frgetvect(source, ch, t0, dur)

                def reduce_500Hz(veca):
                    """Filter and down sample to 500 Hz"""

                    # sampling period
                    Ts = veca[3][0]
                    # sampling frequency
                    fs = 1. / Ts
                    # 10 kHz channels
                    if fs == 10000.:
                        # high pass filtering
                        t_series_10k = signal.sosfilt(sos010k, veca[0])
                        # decimate
                        decim_factor10k = 5
                        t_series_2k = signal.decimate(t_series_10k, decim_factor10k, axis=-1)
                        decim_factor2k = 4
                        t_series_500 = signal.decimate(t_series_2k, decim_factor2k, axis=-1)
                    elif fs == 1000.:
                        # high pass filtering
                        t_series_1k = signal.sosfilt(sos01k, veca[0])
                        # decimate
                        decim_factor1k = 2
                        t_series_500 = signal.decimate(t_series_1k, decim_factor1k, axis=-1)
                    elif fs == 500.:
                        t_series_500 = veca[0]
                    else:
                        raise Exception("Channel {}: Sampling frequency {} Hz not expected".format(ch, fs))
                    return t_series_500

                t_series_500 = reduce_500Hz(vec)

                # units
                units = vec[5]
                t_start = vec[1]
                t_series_f0 = signal.sosfilt(sos0, t_series_500)
                t_series_f1 = signal.sosfilt(sos1, t_series_f0)
                t_series_dec = signal.decimate(t_series_f1, decim_factor, axis=-1)

                # plot filter transfer function if asked
                plot_filter_tf()

                # channel spectrum
                channel = ch[6:11]
                def plot_channel():
                    """Plot original, high and low pass and decimated channel"""

                    nperseg = 2 ** 18

                    plt.figure(channel)
                    f, Pxx = signal.welch(t_series_500, fs=fs, window='hann', detrend='constant', nperseg=nperseg)
                    f_0, P00 = signal.welch(t_series_f0, fs=fs, window='hann', detrend='constant', nperseg=nperseg)
                    f_1, P11 = signal.welch(t_series_f1, fs=fs, window='hann', detrend='constant', nperseg=nperseg)
                    f_dec, Pyy = signal.welch(t_series_dec, fs=fs / decim_factor, window='hann', detrend='constant', nperseg=nperseg / decim_factor)
                    ax = plt.subplot(1, 1, 1)
                    ax.loglog(f, np.sqrt(Pxx), 'b', label='raw')
                    ax.loglog(f_0, np.sqrt(P00), 'c', label='filt0')
                    ax.loglog(f_1, np.sqrt(P11), 'r', label='filt1')
                    ax.loglog(f_dec, np.sqrt(Pyy), 'g', label='dec')
                    ax.grid(which='both', axis='both')
                    ax.set_ylabel(units)
                    # ax.set_ylim([5.e-4, 5.e2])
                    ax.set_xlabel('Frequency Hz')
                    ax.legend()

                # plot_channel()

                def write_channel():
                    """Write original channel in hdf5 file"""

                    print('sa.create_dataset ' + ch)
                    vl = sa.create_dataset(ch, data=vec[0], dtype='float32')
                    vl.attrs['t0'] = t0   # initial GPS time
                    vl.attrs['channel'] = ch
                    vl.attrs['sample_rate'] = fs
                    vl.attrs['units'] = units
                    print(vl)
                    print('Channel {} units: {} sampled at {:6.1f} Hz starting at GPS {}'.format(
                        vl.attrs['channel'],
                        vl.attrs['units'],
                        vl.attrs['sample_rate'],
                        vl.attrs['t0']))
                    with np.printoptions(precision=3, linewidth=160):
                        print(vl[16000:16080:8])

                write_channel()

                # write decimated channel
                name_dec = ch + '_dec'
                dec5 = sa.create_dataset(name_dec, data=t_series_dec, dtype='float32')
                dec5.attrs['t0'] = t0  # initial GPS time
                dec5.attrs['channel'] = name_dec
                dec5.attrs['sample_rate'] = fs / decim_factor
                dec5.attrs['units'] = units
                print(dec5)
                print('Channel {} units: {} sampled at {:6.1f} Hz starting at GPS {}'.format(
                    dec5.attrs['channel'],
                    dec5.attrs['units'],
                    dec5.attrs['sample_rate'],
                    dec5.attrs['t0']))
                with np.printoptions(precision=3, linewidth=160):
                    print(dec5[2000:2010])

    plt.show()
