from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

# Definisci le tue frequenze di risonanza
res_freqs = np.array([0.106, 0.360, 0.482, 0.700, 1.138, 1.373])  # Sostituisci con le tue frequenze

# Definisci il coefficiente di smorzamento
zeta = 0.05  # Sostituisci con il tuo valore di zeta

# Crea un filtro notch per ciascuna frequenza di risonanza
for omega in res_freqs:
    # Converti la frequenza di risonanza in rad/s
    omega_rad = 2 * np.pi * omega
    b, a = signal.iirnotch(omega_rad, zeta)
    # Crea un oggetto di funzione di trasferimento
    sys = signal.TransferFunction(b, a)
    # Calcola la risposta in frequenza
    f = np.linspace(1e-2, 1e1, 10000)
    w = 2 * np.pi * f
    w, h = signal.freqresp(sys, w)
    # Traccia la risposta in frequenza
    plt.figure()
    plt.plot(f, abs(h))
    plt.title('Risposta in frequenza del filtro notch')
    plt.xlabel('Frequenza [Hz]')
    plt.ylabel('Ampiezza')
    plt.grid(True)
    plt.show()
