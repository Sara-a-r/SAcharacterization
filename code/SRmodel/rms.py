import numpy as np
from scipy import linalg
from scipy import signal
import matplotlib.pyplot as plt

# import data
seism = np.loadtxt('../../data/env_ceb_seis_v.txt')                                # data are expressed in m/s^2 (velocity)
freq, Tfnc_1, Tfnc_pl = np.loadtxt('../../data/TFnoControl.txt',unpack=True)   # TF not controlled
_, Tfc_1, Tfc_pl = np.loadtxt('../../data/TF_FSFcontrol.txt',unpack=True)   # TF FSF control

seism = seism * 1.36567e-10
# seismometer ASD
_, Pxx = signal.welch(seism, fs=50., window='hann', nperseg= 2 ** 14)

freq = freq[1:] # remove f = 0 (problems when there is the division)
Pxx = Pxx[1:]

w = 2*np.pi*freq
Pxx = Pxx/w**2  # convert velocity spectrum in displacement

ASDseism = np.sqrt(Pxx)

# compute the response of the controlled system when there is the seism
OutFSF = Tfc_pl[1:] * ASDseism
# cumulated RMS
df = np.diff(freq)
varxx_FSF = np.cumsum(np.flip(df * (OutFSF[0:-1])**2))
rms_FSF = np.flip(np.sqrt(varxx_FSF))

# compute the response of system (not controlled) when there is the seism
Out_nc= Tfnc_pl[1:] * ASDseism
# cumulated RMS
df = np.diff(freq)
varxx_nc = np.cumsum(np.flip(df * (Out_nc[0:-1])**2))
rms_nc = np.flip(np.sqrt(varxx_nc))

plt.title('SR response to seism', size=11)
plt.xlabel('f [Hz]')
plt.ylabel('ASD [m/$\sqrt{Hz}$]')
plt.yscale('log')
plt.xscale('log')
#plt.ylim(1e-7, 1e6)
plt.grid(True)
plt.minorticks_on()



plt.plot(freq, OutFSF, linestyle='-', linewidth=1, marker='', color='blue', label='control')
plt.plot(freq[0:-1], rms_FSF, linestyle='--', linewidth=1, marker='', color='Lime', label='rms control')
plt.plot(freq, Out_nc, linestyle='-', linewidth=1, marker='', color='black', label='no control')
plt.plot(freq[0:-1], rms_nc, linestyle='--', linewidth=1, marker='', color='red', label='rms no control')

plt.legend()

plt.show()