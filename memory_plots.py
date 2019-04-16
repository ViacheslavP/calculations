import sys
sys.path.insert(0, 'core/')
import one_D_scattering as ods
from wave_pack import convolution, delay
from wave_pack import inverse_pulse as pulse
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
ods.RABI = 2
#ods.SINGLE_RAMAN = False
ods.DC = 2#6#7
SHIFT = -2.03#5.04#-0.8#3.56#4.65#4.65#2.80#ods.DC-10.37
freq = np.linspace(-50.5,50.5, 5980)
#Pulse duration
pdTime = 400.#2*np.pi*10
vis = True
noa = 100
sq_reduce = lambda A: np.add.reduce(np.square(np.absolute(A)), axis=1)
ods.RABI_HYP = False


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
#elasticSc.visualize()

#plt.plot(np.arange(0,100,1), ods.rabi_well(np.arange(0,100,1)))
#plt.show()

_envelope = abs(pulse(freq+SHIFT, pdTime))**2

plt.plot(freq, elasticSc.fullTransmittance, 'b-')
plt.plot(freq, elasticSc.fullReflection, 'r-')
plt.plot(freq, elasticSc.SideScattering, 'g-')
#plt.plot(freq, abs(elasticSc.iTransmittance)**2, 'b--')
#plt.plot(freq, abs(elasticSc.iReflection)**2, 'r--')
plt.plot(freq, _envelope/np.amax(_envelope))
plt.show()



freq = freq
_tt, ft = convolution(freq, elasticSc.Transmittance, pulse(freq+SHIFT, pdTime), vg=elasticSc.vg)
_tr, fr = convolution(freq, elasticSc.Reflection, pulse(freq+SHIFT, pdTime), vg=elasticSc.vg)
_ti, fi = convolution(freq, np.ones_like(freq), pulse(freq+SHIFT, pdTime), vg=elasticSc.vg)

t = 25
zi = elasticSc.vg*(t - _ti)
zt = elasticSc.vg*(t - _tt)
zr = -elasticSc.vg*(t - _tr)
zitr = elasticSc.vg*(t + _ti - pdTime)

_rftm = np.empty([len(ft), noa],dtype=np.complex)
_rfrm = np.empty([len(ft), noa],dtype=np.complex)

_rftp = np.empty([len(ft), noa],dtype=np.complex)
_rfrp = np.empty([len(ft), noa],dtype=np.complex)

for k in range(noa):
    _, _rftm[:,k] = convolution(freq, elasticSc.TF2_mm[k,:], pulse(freq+SHIFT, pdTime), vg=elasticSc.vg)
    _, _rfrm[:,k] = convolution(freq, elasticSc.TB2_mm[k,:], pulse(freq+SHIFT, pdTime), vg=elasticSc.vg)

    _, _rftp[:,k] = convolution(freq, elasticSc.TF2_mp[k,:], pulse(freq+SHIFT, pdTime), vg=elasticSc.vg)
    _, _rfrp[:,k] = convolution(freq, elasticSc.TB2_mp[k,:], pulse(freq+SHIFT, pdTime), vg=elasticSc.vg)

_, _rfbb = convolution(freq, elasticSc.TB_pp, pulse(freq+SHIFT, pdTime), vg=elasticSc.vg)
_, _tfbb = convolution(freq, elasticSc.TF_pp, pulse(freq+SHIFT, pdTime), vg=elasticSc.vg)

rfmbb = abs(_rfbb)**2
tfmbb = abs(_tfbb)**2

rfm = sq_reduce(_rfrm)
tfm = sq_reduce(_rftm)
rfp = sq_reduce(_rfrp)
tfp = sq_reduce(_rftp)

rf = rfm*0+rfp+rfmbb
tf = tfm*0+tfp+tfmbb

shifts = np.linspace(-15, 15, 500)
ir, tr = delay(freq, shifts, elasticSc.Reflection, pdTime, elasticSc.vg)
it, tt = delay(freq, shifts, elasticSc.Transmittance, pdTime, elasticSc.vg)
plt.plot(shifts, tt-pdTime/2, 'b-')
plt.plot(shifts, tr-pdTime/2, 'r-')

plt.fill_between(shifts, 0, ir, alpha = 0.3, facecolor = 'red')
plt.fill_between(shifts, 0, it, alpha = 0.3, facecolor = 'blue')
plt.show()

plt.plot(zi, abs(fi)**2, 'g-', label="Initial")
plt.plot(zitr, abs(fi)**2, 'm-', label="Initial, time reversed")
plt.plot(zt, abs(ft)**2, 'b-', label="Transmitted,elastic")
plt.plot(zr, abs(fr)**2, 'r-', label="Reflected,elastic")
plt.plot(zt, tf, 'b--', label="Transmitted,Raman")
plt.plot(zr, rf, 'r--', label="Reflected,Raman")
#plt.legend()
plt.show()

"""
Guess: $\frac{d}{d\omega} arg t \approx \tau$
"""

plt.plot(freq, np.angle(elasticSc.Transmittance))
#plt.legend()
plt.show()

