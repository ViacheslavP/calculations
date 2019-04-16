import sys
sys.path.insert(0, 'core/')
import one_D_scattering as ods
from wave_pack import convolution, delay
from wave_pack import ideal_spectra as isp
from wave_pack import inverse_pulse as pulse
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
ods.RABI = 2
#ods.SINGLE_RAMAN = False
ods.DC = 0#-2#6#7
SHIFT = 0#2.42#5.04#-0.8#3.56#4.65#4.65#2.80#ods.DC-10.37
raz = 1000980
freq = np.linspace(-180.5,180.5, raz)
#Pulse duration
pdTime = 0.01#26.7#26.9#2*np.pi
vis = True
noa = 300
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



#elasticSc = ods.ensemble(**args)
#elasticSc.generate_ensemble()
#elasticSc.visualize()

#plt.plot(np.arange(0,100,1), ods.rabi_well(np.arange(0,100,1)))
#plt.show()

_envelope = abs(pulse(freq+SHIFT, pdTime))**2

#plt.plot(freq, elasticSc.fullTransmittance, 'b-')
#plt.plot(freq, elasticSc.fullReflection, 'r-')
#plt.plot(freq, elasticSc.SideScattering, 'g-')
#plt.plot(freq, abs(elasticSc.iTransmittance)**2, 'b--')
#plt.plot(freq, abs(elasticSc.iReflection)**2, 'r--')
#plt.plot(freq, _envelope/np.amax(_envelope))
#plt.show()


#Two side g_b

args_isp = {'omega' : freq,
            'gamma' : 10,
            'rabi'  : 2,
            'rabidt': 0,
            'gamma_parasite':0}

chain = isp(**args_isp)
fig = plt.figure()

plt.subplot(1, 2, 1)
plt.plot(freq, abs(chain.transmission())**2, 'b-')
plt.plot(freq, abs(chain.reflection())**2, 'r-')
plt.plot(freq, np.angle(chain.double_entry()), 'm-')

freq = freq
_, ft_f = convolution(freq, chain.transmission(), pulse(freq+SHIFT, pdTime), vg=0.7)
_, fr_f = convolution(freq, chain.reflection(), pulse(freq+SHIFT, pdTime), vg=0.7)
_, fd_f = convolution(freq, chain.double_entry(), pulse(freq+SHIFT, pdTime), vg=0.7)

#_tt, ft_b = convolution(freq, (elasticSc.Transmittance), pulse(freq+SHIFT, pdTime), vg=elasticSc.vg)
#_tr, fr_b = convolution(freq, (elasticSc.Reflection), pulse(freq+SHIFT, pdTime), vg=elasticSc.vg)

_ti, fi_f = convolution(freq, np.ones_like(freq), pulse(freq+SHIFT, pdTime), vg=0.7)
_ti, fi_b = convolution(freq, np.zeros_like(freq), pulse(freq+SHIFT, pdTime), vg=0.7)

ft = ft_f#+fr_b
fr = fr_f#+fr_f

t = 10
zi_f = (t + _ti)
zi_b = (t + _ti)



zitr = (t + _ti - pdTime)
plt.subplot(1, 2, 2)
plt.plot(zi_f, abs(fi_f)**2, 'g-', label="Initial")
plt.plot(zi_b, abs(fi_b)**2, 'g--', label="Initial")

plt.plot(zi_f, abs(ft)**2, 'b-', label="Transmitted")
plt.plot(zi_f, abs(fd_f)**2, 'm-', label="DoubleEntry")
plt.plot(zi_b, abs(fr)**2, 'r-', label="Reflected")

plt.show()

"""

zib = np.empty(raz)
zib[0:raz-2] = zi_b[0:raz-2]
zif = np.empty(raz)
zif[0:raz-2] = zi_f[0:raz-2]
idx = 5490
print(idx)

fib = np.empty(raz, dtype=np.complex)
fif = np.empty(raz, dtype=np.complex)

nm = 200
pi = np.empty([nm,nm], dtype=np.float32)
pdTimes = np.linspace(30, 40, nm)
shifts = np.linspace(2, 2.5, nm)
for i in range(1):
    for j in range(1):
        pt = pdTimes[i]
        shift = shifts[j]

        _tt, ft_f = convolution(freq, elasticSc.Transmittance, f_pulse(freq + shift, pt), vg=elasticSc.vg)
        _tr, fr_f = convolution(freq, elasticSc.Reflection, f_pulse(freq + shift, pt), vg=elasticSc.vg)

        _tt, ft_b = convolution(freq, np.conj(elasticSc.Transmittance), b_pulse(freq + shift, pt), vg=elasticSc.vg)
        _tr, fr_b = convolution(freq, 1 * np.conj(elasticSc.Reflection), b_pulse(freq + shift, pt), vg=elasticSc.vg)

        _ti, fi_f = convolution(freq, np.ones_like(freq), f_pulse(freq + shift, pt), vg=elasticSc.vg)
        _ti, fi_b = convolution(freq, np.ones_like(freq), b_pulse(freq + shift, pt), vg=elasticSc.vg)

        ft = ft_f + fr_b
        fr = ft_b + fr_f



        pi[i,j] = np.trapz(abs(fr[0:idx])**2, zi_b[0:idx])/np.trapz(abs(fi_f[0:raz-2])**2, zi_b[0:raz-2])


x,y = np.meshgrid(pdTimes, shifts)

plt.pcolor(x,y,pi, cmap='RdBu', vmin=np.amin(pi), vmax=np.amax(pi))
plt.colorbar()
plt.show()
print(np.amax(pi))
_am = np.argmax(pi)
ij = np.unravel_index(_am, pi.shape)
print(pdTimes[ij[0]], shifts[ij[1]])


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(x,y,pi)
plt.show()

plt.plot(abs(fr)**2)
plt.plot(abs(fi_f)**2)
plt.show()

"""