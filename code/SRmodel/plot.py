import numpy as np
from scipy import linalg
from scipy import signal
import matplotlib.pyplot as plt

# import data
data = np.loadtxt('LVDT_t0')

plt.plot(data, linestyle='-', linewidth=1, marker='', label='seismometer')
plt.show()