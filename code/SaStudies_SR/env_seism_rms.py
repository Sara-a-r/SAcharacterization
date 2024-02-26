import numpy as np
from scipy import linalg
from scipy import signal
import matplotlib.pyplot as plt

# import data
data = np.loadtxt('env_ceb_seis_v.txt')     # data are expressed in m/s^2 (velocity)

# evaluate PSD (spectrum of seismometer)
f, Pxx = signal.welch(data, fs=50., window='hann', nperseg= 2 ** 14)

f = f[1:] # remove f = 0 (problems when there is the division)
Pxx = Pxx[1:]

w = 2*np.pi*f
Pxx = Pxx/w**2  # convert velocity spectrum in displacement

# cumulated RMS
df = np.diff(f)
varxx = np.cumsum(np.flip(df * Pxx[0:-1]))

plt.title('ASD seismometer channel', size=11)
plt.xlabel('f [Hz]')
plt.ylabel('ASD [m/$\sqrt{Hz}$]')
plt.yscale('log')
plt.xscale('log')
plt.grid(True)
plt.minorticks_on()



plt.plot(f, np.sqrt(Pxx), linestyle='-', linewidth=1, marker='', color='blue', label='seismometer')
plt.plot(f[0:-1], np.flip(np.sqrt(varxx)), linestyle='-', linewidth=1, marker='', color='Lime', label='rms')
plt.legend()

plt.show()