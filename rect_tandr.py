import sys
sys.path.insert(0, 'core/')
import one_D_scattering as ods
from wave_pack import convolution, rect_pulse
import numpy as np
from matplotlib import pyplot as plt

import ctypes
mkl_rt = ctypes.CDLL('libmkl_rt.so')
mkl_get_max_threads = mkl_rt.mkl_get_max_threads
def mkl_set_num_threads(cores):
    mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(cores)))

mkl_set_num_threads(4)
print(mkl_get_max_threads()) # says 4

"""
Control field and calculation parameters:
"""

ods.PAIRS = False
ods.RABI = 50.
#ods.SINGLE_RAMAN = False
ods.DC = -50
SHIFT = ods.DC-10.37
freq = np.linspace(-103.5,103.5, 980)+SHIFT
#Pulse duration
pdTime = 2*np.pi

args = {

    'nat': 20,  # number of atoms
    'nb': 0,  # number of neighbours in raman chanel (for L-atom only)
    's': 'chain',  # Stands for atom positioning : chain, nocorrchain and doublechain
    'dist': 0.,  # sigma for displacement (choose 'chain' for gauss displacement.)
    'd': 1.5,  # distance from fiber
    'l0': 2.0/2,  # mean distance between atoms (in lambda_m /2 units)
    'deltaP': freq,  # array of freq.
    'typ': 'L',  # L or V for Lambda and V atom resp.
    'ff': 0.3
    }


elasticSc = ods.ensemble(**args)
elasticSc.generate_ensemble()
elasticSc.visualize()

elasticSc = ods.ensemble(**args)
elasticSc.generate_ensemble()
elasticSc.visualize()

freq = freq-SHIFT
_tt, _ft = convolution(freq, elasticSc.Transmittance, rect_pulse(freq, pdTime), vg=elasticSc.vg)
_tr, _fr = convolution(freq, elasticSc.Reflection, rect_pulse(freq, pdTime), vg=elasticSc.vg)
_ti, _fi = convolution(freq, np.ones_like(freq), rect_pulse(freq, pdTime), vg=elasticSc.vg)

t = 15
zi = elasticSc.vg*(t - _ti)
zt = elasticSc.vg*(t - _tt)
zr = -elasticSc.vg*(t - _tr)
zitr = elasticSc.vg*(-t - _ti + pdTime)


ods.PAIRS = True
print("Pairs")
elasticSc = ods.ensemble(**args)
elasticSc.generate_ensemble()
elasticSc.visualize()


_tt, _ftf = convolution(freq, elasticSc.Transmittance, rect_pulse(freq, pdTime), vg=elasticSc.vg)
_tr, _frf = convolution(freq, elasticSc.Reflection, rect_pulse(freq, pdTime), vg=elasticSc.vg)
_ti, _fif = convolution(freq, np.ones_like(freq), rect_pulse(freq, pdTime), vg=elasticSc.vg)


plt.plot(zi, abs(_fi)**2, 'g-', label="Initial")
plt.plot(zitr, abs(_fi)**2, 'm-', label="Initial, time reversed")
plt.plot(zt, abs(_ft)**2, 'b-', label="Transmitted")
plt.plot(zr, abs(_fr)**2, 'r-', label="Reflected")

plt.plot(zi, abs(_fif)**2, 'g--', label="Initial")
plt.plot(zitr, abs(_fif)**2, 'm--', label="Initial, time reversed")
plt.plot(zt, abs(_ftf)**2, 'b--', label="Transmitted")
plt.plot(zr, abs(_frf)**2, 'r--', label="Reflected")

#plt.legend()
plt.show()