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
    # t0 = 1374011418  # 21 Jul 2023 21:50:00 UTC
    t0 = 1390626064  # 30 Jan 2024  5:00:00 UTC

    dur = 1800
    source = '/virgoData/ffl/rds.ffl'

    # towers
    towers = ['SR']
    # buildings
    buildings = {'SR': 'CEB'}
    # building environmental channels
    env = {'CEB': 'V1:ENV_CEB_SEIS_V_50Hz'}
    # seismic filters
    sfilters = ['F0', 'F1', 'F2', 'F3', 'F4', 'F7']
    # vertical distance
    lvdt = {'F0': '_LVDT_V_50Hz',
            'F1': '_LVDT_V_50Hz',
            'F2': '_LVDT_V_50Hz',
            'F3': '_LVDT_V_50Hz',
            'F4': '_LVDT_V_50Hz',
            'F7': '_LVDT_V_50Hz'}

    # accelerometers
    f0accel = ['CRBAR1', 'CRBAR2', 'CRBAR']
    accel = {'CRBAR1': '_F0_ACC_V1_50Hz', 'CRBAR2': '_F0_ACC_V2_50Hz'}
    # others = {'F7LV1': '_F7_LVDT_V1', 'F7LV2': '_F7_LVDT_V2', 'F7LV3': '_F7_LVDT_V3'}
    corrsa = {'F0': 'F0_Y_CORR_50Hz'}
    br = ['_BR_LVDT1_50Hz', '_BR_LVDT2_50Hz', '_BR_LVDT3_50Hz', '_BR_PZ_V1_50Hz', '_BR_PZ_V2_50Hz', '_BR_PZ_V3_50Hz',
          '_BR_Y_50Hz', '_BR_Y_BL_50Hz']
    f0others = ['_F0_ACC_V1_FB_50Hz', '_F0_ACC_V2_FB_50Hz', '_F0_COIL_V1_50Hz', '_F0_COIL_V2_50Hz',
                '_F0_y1_y7_50Hz', '_F0_ySA2_50Hz']
    sc = ['_F7_Y_50Hz']

    for tt in towers:
        chlist = []
        # environmental channels
        chlist.append(env[buildings[tt]])
        # filter lvdt channels
        for sf in sfilters:
            chlist.append('V1:Sa_' + tt + '_' + sf + lvdt[sf])
        # accelerometers
        for acc in accel:
            chlist.append('V1:Sa_' + tt + accel[acc])
        for corr in corrsa:
            chlist.append('V1:Sa_' + tt + '_' + corrsa[corr])
        for brch in br:
            chlist.append('V1:Sa_' + tt + brch)
        for f0othersch in f0others:
            chlist.append('V1:Sa_' + tt + f0othersch)
        for scch in sc:
            chlist.append('V1:Sc_' + tt + scch)

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

                # units
                units = vec[5]
                t_start = vec[1]
                t_series = vec[0]
                fs = 50.

                # channel spectrum
                channel = ch[6:11]

                def plot_channel():
                    """Plot original, high and low pass and decimated channel"""

                    nperseg = 2 ** 14

                    plt.figure(channel)
                    f, pxx = signal.welch(t_series, fs=fs, window='hann', detrend='constant', nperseg=nperseg)
                    ax = plt.subplot(1, 1, 1)
                    ax.loglog(f, np.sqrt(pxx), 'b', label='raw')
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
                    vl.attrs['t0'] = t_start   # initial GPS time
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
                plot_channel()

    plt.show()
