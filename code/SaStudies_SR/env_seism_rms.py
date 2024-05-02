import numpy as np
from scipy import linalg
from scipy import signal
import matplotlib.pyplot as plt

# import data
data = np.loadtxt('../../data/env_ceb_seis_v.txt')     # data are expressed in m/s^-2
data = data * 1.36567e-10

# evaluate PSD (spectrum of seismometer)
f, Pxx = signal.welch(data, fs=50., window='hann', nperseg= 2 ** 14)

#save frequency array
#np.savetxt('../../data/freq.txt', f, header='f[Hz]')


f = f[1:] # remove f = 0 (problems when there is the division)
Pxx = Pxx[1:]

w = 2*np.pi*f
Pxx = Pxx/w**2  # convert seism spectrum in displacement

# cumulated RMS
df = np.diff(f)
varxx = np.cumsum(np.flip(df * Pxx[0:-1]))  #cumulative variance
rms = np.flip(np.sqrt(varxx))

plt.title('v seism Virgo', size=13)
plt.xlabel('Frequency [Hz]',size=12)
plt.ylabel('ASD [m/$\sqrt{Hz}$]',size=12)
plt.yscale('log')
plt.xscale('log')
plt.xlim(0.05,20)
plt.ylim(10**-10,3*10**-6)
plt.grid(True, which='both',ls='-', alpha=0.3, lw=0.5)
plt.minorticks_on()



plt.plot(f, np.sqrt(Pxx), linestyle='-', linewidth=1, marker='', label='seismometer')
#plt.plot(f[0:-1], rms , linestyle='-', linewidth=1, marker='', color='Lime', label='rms')
plt.legend()

plt.show()