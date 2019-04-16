import sys
sys.path.insert(0, 'core/')
import one_D_scattering as ods
from wave_pack import convolution, delay
from wave_pack import inverse_pulse as pulse
from wave_pack import efficiency
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import cumtrapz as ct

import ctypes
mkl_rt = ctypes.CDLL('libmkl_rt.so')
mkl_get_max_threads = mkl_rt.mkl_get_max_threads
def mkl_set_num_threads(cores):
    mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(cores)))

mkl_set_num_threads(4)
print(mkl_get_max_threads()) # says 4

"""
Control field and calculation parameters:
87%
for 100 atoms
Rabi 2
SHIFT -2.37
pdTime 14.5

18%
for 10 atoms
Rabi 2
SHIFT -2.38
pdTime 47
"""

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


ods.PAIRS = False
ods.RABI = 0
#ods.SINGLE_RAMAN = False
ods.DC = 0*-6#6#7
SHIFT = 0#5.04#-0.8#3.56#4.65#4.65#2.80#ods.DC-10.37
raz = 8000
freq = np.linspace(-80.5,80.5, raz)
#Pulse duration
pdTime = 0.418#26.7#26.9#2*np.pi
vis = True
noa = 10
sq_reduce = lambda A: np.add.reduce(np.square(np.absolute(A)), axis=1)
ods.RABI_HYP = False
ods.RADIATION_MODES_MODEL = 1
ods.VACUUM_DECAY = 1
"""
ods.PAIRS = False
ods.RABI = 2
#ods.SINGLE_RAMAN = False
ods.DC = -6#6#7
SHIFT = 6.15#5.04#-0.8#3.56#4.65#4.65#2.80#ods.DC-10.37
raz = 8000
freq = np.linspace(-80.5,80.5, raz)
#Pulse duration
pdTime = 15#26.7#26.9#2*np.pi
vis = True
noa = 300
sq_reduce = lambda A: np.add.reduce(np.square(np.absolute(A)), axis=1)
ods.RABI_HYP = False
"""

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
plt.plot(freq, abs(elasticSc.iTransmittance)**2, 'b--')
plt.plot(freq, abs(elasticSc.iReflection)**2, 'r--')
plt.plot(freq, _envelope/np.amax(_envelope))
plt.plot(freq, (np.ones_like(freq) - abs(elasticSc.Reflection-elasticSc.Transmittance)**2)/4, 'm--')
plt.xlim(left=-15, right=5)
plt.show()

#Two side g_b

freq = freq
_tt, ft_f = convolution(freq, elasticSc.Reflection-elasticSc.Transmittance, pulse(freq+SHIFT, pdTime), vg=elasticSc.vg)
_tt, ftasym_f = convolution(freq, -elasticSc.Transmittance, pulse(freq+SHIFT, pdTime), vg=elasticSc.vg)
_ti, fi_f = convolution(freq, np.ones_like(freq), pulse(freq+SHIFT, pdTime), vg=elasticSc.vg)
ft = ft_f

plt.plot(_ti, abs(fi_f)**2, 'g-', label="Initial")
plt.plot(_ti, abs(ft_f)**2, 'b-', label="Transmitted/Reflected (symmetric)")
plt.plot(_ti, abs(ftasym_f)**2, 'b--', label="Transmitted/Reflected (symmetric)")
plt.xlim(left=-10, right=20)
plt.show()


fib = np.empty(raz, dtype=np.complex)
fif = np.empty(raz, dtype=np.complex)

nm = 100
ntimes = 200
idx = 4000
pi = np.empty([ntimes,nm], dtype=np.float32)
pdTimes = np.linspace(0.1, 0.7, ntimes)
pdGammas = np.linspace(0.001, 20.2, ntimes)
shifts = np.linspace(-30, 30, nm)
psi = np.empty(ntimes, dtype=np.float32)
fidelity = np.empty(ntimes, dtype=np.float32)


for i in range(ntimes):
    _tt, ft = convolution(freq, elasticSc.Reflection-elasticSc.Transmittance, pulse(freq + SHIFT, pdTimes[i]), vg=elasticSc.vg)
    #_ti, fi = convolution(freq, np.ones_like(freq), pulse(freq + shifts[j], pdTimes[i]), vg=elasticSc.vg)
    psi[i] = +abs(np.trapz((abs(ft[idx::]) ** 2), x=_tt[idx::]))
    _foo = (elasticSc.Reflection-elasticSc.Transmittance)*pulse(freq + SHIFT, pdTimes[i])*pulse(freq + SHIFT, pdTimes[i])
    _foi = pulse(freq + SHIFT, pdTimes[i])*pulse(freq + SHIFT, -pdTimes[i])
    fidelity[i] = efficiency(pdTimes[i], _tt, ft) #abs(np.trapz(_foo, x=freq)) / abs(np.trapz(_foi, x=freq))

#plt.plot(pdTimes, psi)
#plt.show()
plt.plot(pdTimes, fidelity)

plt.show()

"""
for i in range(ntimes):
    for j in range(nm):

        _tt, ft = convolution(freq, elasticSc.Reflection-elasticSc.Transmittance, pulse(freq + shifts[j], pdTimes[i]), vg=elasticSc.vg)
        #_ti, fi = convolution(freq, np.ones_like(freq), pulse(freq + shifts[j], pdTimes[i]), vg=elasticSc.vg)
        pi[i, j] = -abs(np.trapz((abs(ft[idx::]) ** 2), x=_tt[idx::])) / pdTimes[i]


x, y = np.meshgrid(shifts, pdTimes)

plt.pcolor(x, y, pi, cmap='RdBu', vmin=np.amin(pi), vmax=np.amax(pi))
plt.colorbar()
plt.show()

print(np.amax(pi))
_am = np.argmax(pi)
ij = np.unravel_index(_am, pi.shape)
print(pdTimes[ij[0]], shifts[ij[1]])


from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.plot_surface(x,y,pi)
#plt.show()

"""