import numpy as np
from scipy.optimize import broyden1

import matplotlib
font = {'family' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)

import matplotlib.pyplot as plt

pseudoRabi = lambda rabi, detun: 1 / np.sqrt(2) * np.sqrt(detun ** 2 + rabi ** 2 - 1/4 \
                        + np.sqrt(detun ** 4 + (2* rabi ** 2 + 1/2) * detun ** 2 + rabi ** 4 - 0.5 * rabi ** 2 + 1))
width = lambda om, detuning: 0.5 * (om - detuning)/(2*om - detuning)

detunings = np.linspace(-20, 20, 1000)
rabis = [5.]
nrabis = len(rabis); ndetun = len(detunings)
omegas_minus = np.empty([nrabis, ndetun], dtype=np.float32)
omegas_plus = np.empty([nrabis, ndetun], dtype=np.float32)
width_minus = np.empty([nrabis, ndetun], dtype=np.float32)
width_plus = np.empty([nrabis, ndetun], dtype=np.float32)

for i in range(nrabis):
    for j in range(ndetun):
        omegas_minus[i, j] = 0.5 * detunings[j] - 0.5 * pseudoRabi(rabis[i], detunings[j])
        omegas_plus[i, j] = 0.5 * detunings[j] + 0.5 * pseudoRabi(rabis[i], detunings[j])
        width_minus[i, j] = width(omegas_minus[i,j], detunings[j])
        width_plus[i, j] = width(omegas_plus[i,j], detunings[j])



for i in range(nrabis):
    plt.plot(detunings, omegas_minus[i, :])
    plt.plot(detunings, omegas_plus[i, :])

plt.plot(detunings, np.zeros_like(detunings))
plt.plot(detunings, detunings)
plt.show()

for i in range(nrabis):
    plt.plot(detunings, width_minus[i, :])
    plt.plot(detunings, width_plus[i, :])


plt.show()
