import sys
sys.path.insert(0, 'core/')
import one_D_scattering as ods
from wave_pack import convolution, delay
from wave_pack import gauss_pulse_v2 as pulse
import numpy as np
from matplotlib import pyplot as plt
sq_reduce = lambda A: np.add.reduce(np.square(np.absolute(A)), axis=1)


"""
Control field and calculation parameters:
"""
ods.RABI_HYP = False
ods.PAIRS = False
ods.RABI = 1 # Rabi frequency
ods.DC = 0# Control field detuning
SHIFT = 0# Carrier frequency of a photon
freq = np.linspace(-20.5,20.5, 1980)
#Pulse duration
pdTime = 10 # pulse time, gammma^-1

noa = 100 #Number of atoms




args = {
    'nat': noa,  # number of atoms
    'nb': 0,  # number of neighbours in raman chanel (for L-atom only)
    's': 'chain',  # Stands for atom positioning : chain, nocorrchain and doublechain
    'dist': 0.,  # sigma for displacement (choose 'chain' for gauss displacement.)
    'd': 1.5,  # distance from fiber
    'l0': 2.00/2,  # mean distance between atoms (in lambda_m /2 units)
    'deltaP': freq,  # array of freq.
    'typ': 'L',  # L or V for Lambda and V atom resp.
    'ff': 0.3
    }



elasticSc = ods.ensemble(**args)
elasticSc.generate_ensemble()

_, ft = convolution(freq, elasticSc.Transmittance, pulse(freq+SHIFT, pdTime))
_, fr = convolution(freq, elasticSc.Reflection, pulse(freq+SHIFT, pdTime))
ti, fi = convolution(freq, np.ones_like(freq), pulse(freq+SHIFT, pdTime))



plt.plot(ti, abs(fi)**2, 'g--', label="Initial")
plt.plot(ti, abs(fr)**2, 'r-', label="Reflected")
plt.plot(ti, abs(ft)**2, 'b-', label="Transmitted")
plt.legend()
plt.show()


