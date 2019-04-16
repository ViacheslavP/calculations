import sys
sys.path.insert(0, 'core/')
import numpy as np
from bmode import exact_mode
from matplotlib import pyplot as plt

def Sus(freqs, gamma):
    result = np.empty_like(freqs, dtype=np.complex)
    result = -1./(freqs + 0.5j*gamma)
    return result



a = 2 * np.pi * 200 / 780
ndots = 2
nof = 100
width = 3

c = 1
kv = 1
d00 = 0.5

argmode = {'omega': 1,
        'n': 1.452,
        'a': a
        }
m = exact_mode(**argmode)

x = np.linspace(1.5, 2., ndots) * a
freq = np.linspace(-width, width,  nof)

em = -m.E(x)
ez = m.Ez(x)
ep = m.dE(x)


gammaFull = np.zeros([ndots], dtype=complex)
for i in range(ndots):
    emfi = np.array([ep[i], ez[i], em[i]], dtype=complex)
    emfic = np.conjugate(np.array([ep[i], ez[i], em[i]], dtype=complex))

    gammaFull[i] = 1 + 4 * d00 * d00 * np.real(
    (np.dot(emfi, emfic) * 2 * np.pi * kv * (1 / m.vg - 1 / c)))

gammma = gammaFull[0]
plt.plot(freq, d00*d00*np.real(Sus(freq, gammma)))
plt.plot(freq, d00*d00*np.imag(Sus(freq, gammma)))

plt.show()