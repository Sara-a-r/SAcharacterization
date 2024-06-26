import numpy as np
from scipy import linalg
from scipy import signal
import matplotlib.pyplot as plt
import h5py

#----------------------------PSD plot function-------------------#
def psd_plots(frvect, ch, nperseg, iplot):

    # save data LVDT not filtered (in time domain)
    #np.savetxt('LVDT_t' + '{}'.format(iplot), frvect[ch][2000:])

    #-----------------digital filter----------------#
    #low pass filter (for the first normal frequency)
    high = 0.2
    order = 4
    b, a = signal.butter(order, Wn=high, btype='lowpass', output='ba', fs=62.5)
    data_filtered = signal.filtfilt(b, a, frvect[ch][2000:])

    f, Pxx = signal.welch(data_filtered, fs=frvect[ch].attrs['sample_rate'], window='hann', nperseg=nperseg)

    # spectrum
    ax = plt.subplot(3,3,iplot+1)
    # cumulated RMS
    df = np.diff(f)
    varxx = np.cumsum(np.flip(df * Pxx[0:-1]))
    ax.loglog(f, np.sqrt(Pxx))

    # save data (psd LVDT)
    #dataLVDT = np.column_stack((f,np.sqrt(Pxx)))
    #np.savetxt('LVDT' + '{}'.format(iplot), dataLVDT)

    ax.loglog(f[0:-1], np.flip(np.sqrt(varxx)))
    ax.grid()
    chname = frvect[ch].attrs['channel']
    ax.set_ylabel(chname[9:18] + ' ' + frvect[ch].attrs['units'])
    yl = ax.get_ylim()
    yl1 = [5.e-4, yl[1]]
    ax.set_ylim(yl1)

    ax.set_xlabel('Frequency Hz')
    axLVDT.loglog(f, np.sqrt(Pxx) * 100.**(-iplot), label=chname)

#--------------------------Covariance matrix-------------------------#

def cov(X):
    """
    Covariance matrix
    note: specifically for mean-centered data
    note: numpy's `cov` uses N-1 as normalization
    """

    return np.dot(X.T, X) / X.shape[0]

#---------------------------PCA analysis-----------------------#
def pca(data, pc_count = None):
    """
    Principal component analysis using eigenvalues
    note: this mean-centers and auto-scales the data (in-place)
    """
    data -= np.mean(data, 0)
    data /= np.std(data, 0)
    C = cov(data)
    E, V = linalg.eigh(C)
    key = np.argsort(E)[::-1][:pc_count]
    E, V = E[key], V[:, key]
    U = np.dot(data, V)  # used to be dot(V.T, data.T).T
    return U, E, V

#--------------------------Channels---------------------------#
#channels = ['V1:Sa_NE_F0_LVDT_V_500Hz', 'V1:Sa_NE_F1_LVDT_V_500Hz', 'V1:Sa_NE_F2_LVDT_V_500Hz',
#            'V1:Sa_NE_F3_LVDT_V_500Hz', 'V1:Sa_NE_F4_LVDT_V_500Hz', 'V1:Sa_NE_F7_LVDT_V_500Hz']

channels = ['V1:Sa_NE_F0_LVDT_V_500Hz','V1:Sa_NE_F1_LVDT_V_500Hz', 'V1:Sa_NE_F2_LVDT_V_500Hz',
            'V1:Sa_NE_F3_LVDT_V_500Hz', 'V1:Sa_NE_F4_LVDT_V_500Hz']                 #not considering F7

othchans = ['V1:ENV_NEB_SEIS_V', 'V1:Sa_NE_F7_LVDT_V1', 'V1:Sa_NE_F7_LVDT_V1', 'V1:Sa_NE_F7_LVDT_V2',
            'V1:Sc_NE_MAR_PSDM_V1', 'V1:Sc_NE_MAR_PSDM_V2',
            'V1:Sc_NE_MAR_PSDT_V1', 'V1:Sc_NE_MAR_PSDT_V2',
            'V1:Sc_NE_MIR_PSDF_V1', 'V1:Sc_NE_MIR_PSDF_V2',
            'V1:Sa_NE_F0_ACC_V1', 'V1:Sa_NE_F0_ACC_V2', 'V1:Sa_NE_F0_ACC_Y_500Hz', 'V1:Sa_NE_F0_Y_500Hz',
            'V1:Sa_NE_F0_Y_CORR_500Hz', 'V1:Sc_NE_MIR_Y_AA', 'V1:Sc_NE_txAA']


if __name__ == '__main__':

    with (h5py.File("SaNE20230828.hdf5", "r") as fSa):      #read data and do all plots
        sa = fSa['NE']   #data

        #----------------------LVDT psd-----------------------#

        figLVDT = plt.figure('LVDT')        #plot all LVDT data
        axLVDT = figLVDT.subplots()
        plt.figure('LVDT separated')        #plot LVD separated

        iplot = 0           #index that runs on channels of LVDT
        for ch in channels:
            print(ch)
            ch_dec = ch + '_dec'
            nperseg = 2 ** 16
            psd_plots(sa, ch_dec, nperseg, iplot)       #psd of each LVDT
            iplot += 1


        psd_plots(sa, 'V1:ENV_NEB_SEIS_V_dec', nperseg, iplot)      #add the psd for env_seis channel
        iplot += 1

        axLVDT.grid(which='both', axis='both')
        axLVDT.set_xlim([1.e-2, 5.e0])
        axLVDT.set_xlabel('Frequency Hz')
        yl = axLVDT.get_ylim()
        yl1 = [5.e-14, yl[1]]
        axLVDT.set_ylim(yl1)

        axLVDT.legend()

        #-------------------------PCA analysis---------------------#                    #nota: qui alla funzione pca vengono passati i dati in
        print('Principal Component Analysis')                                           #funzione del tempo, invece nel codice test i dati dopo
                                                                                        #aver fatto la psd (dei dati già filtrati)
        # --------digital low pass filter---------#
        high = 0.2
        order = 4
        b, a = signal.butter(order, Wn=high, btype='lowpass', output='ba', fs=62.5)
        data_filtered = []
        for channel in channels:
            data_filtered_ch = signal.filtfilt(b, a, sa[channel+'_dec'][2000:])
            data_filtered.append(data_filtered_ch)

        vpos = np.array([i for i in data_filtered]).T
        #vpos = np.array([sa[channel+'_dec'][2000:] for channel in channels]).T
        print(vpos)
        U, E, V = pca(vpos, 6)
        with np.printoptions(precision=3, linewidth=160):
            print('U')
            print(U)
            print('E')
            print(E)
            print('V')
            print(V)

        fs = sa[ch_dec].attrs['sample_rate']    #sample rate (62.5Hz)
        units = sa[ch_dec].attrs['units']

        figPCA = plt.figure('PCA')
        axPCA = figPCA.subplots()
        plt.figure('PCA separated')
        iplot = 0

        for iU in range(len(E)):
            f, Pxx = signal.welch(U[:,iU], fs=fs, window='hann', nperseg=nperseg)

            # spectrum
            ax = plt.subplot(2, 3, iU + 1)
            # cumulated RMS
            df = np.diff(f)
            varxx = np.cumsum(np.flip(df * Pxx[0:-1]))
            ax.loglog(f, np.sqrt(Pxx))                 #figure PCA separated

            #save data
            #dataPCA = np.column_stack((f, np.sqrt(Pxx)))
            #np.savetxt('PCA' + '{}'.format(iU), dataPCA)

            ax.loglog(f[0:-1], np.flip(np.sqrt(varxx)))
            ax.grid()
            chname = 'U' + '{}'.format(iU)
            ax.set_ylabel(chname + ' ' + units)
            ax.set_ylim([5.e-5, 5.e0])
            ax.set_xlabel('Frequency Hz')

            axPCA.loglog(f, np.sqrt(Pxx) * 100.**(-iU), label='U' + '{}'.format(iU))    #figure PCA

        axPCA.grid(which='both', axis='both')
        # ax.set_ylim([5.e-4, 5.e2])
        axPCA.set_xlim([1.e-2, 5.e0])
        yl = axPCA.get_ylim()
        yl1 = [5.e-15, yl[1]]
        axPCA.set_ylim(yl1)
        axPCA.set_xlabel('Frequency Hz')
        axPCA.legend()

        #------------------seism transfer functions--------------#
        chseism = 'V1:ENV_NEB_SEIS_V_dec'
        seism = sa[chseism][2000:]
        list = ['0', '1', '2', '3', '4', '7']
        i = 0
        for ch in channels:
            ch_dec = ch + '_dec'
            figTF = plt.figure('TF ' + ch_dec + '/' + 'chseism')
            f, Pxy = signal.csd(seism, sa[ch_dec][2000:], fs=fs, window='hann', nperseg=nperseg)
            Pxy = Pxy / (1.j * 2. * np.pi * f)
            ###
            data = np.column_stack((f,np.sqrt(np.abs(Pxy))))
            #np.savetxt('TF ' + list[i], data)
            i = i+1
            ####
            f, Cxy = signal.coherence(seism, sa[ch_dec][2000:], fs=fs, window='hann', nperseg=nperseg)
            axTF = figTF.subplots(3,1)
            axTF[0].loglog(f, np.sqrt(np.abs(Pxy)))
            axTF[1].semilogx(f, np.angle(Pxy))
            axTF[2].semilogx(f, Cxy)
            yl = axTF[0].get_ylim()
            yl1 = [1.e-3, yl[1]]
            axTF[0].set_ylim(yl1)
            axTF[0].grid(which='both', axis='both')
            axTF[0].set_xlim([1.e-1, 3.e0])
            axTF[1].grid(which='both', axis='both')
            axTF[1].set_xlim([1.e-1, 3.e0])
            axTF[1].set_ylim([-3.5, 3.5])
            axTF[2].grid(which='both', axis='both')
            axTF[2].set_xlim([1.e-1, 3.e0])
            axTF[2].set_ylim([0., 1.])
            axTF[2].set_xlabel('Frequency Hz')

    plt.show()
