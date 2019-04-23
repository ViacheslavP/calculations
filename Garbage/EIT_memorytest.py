import sys
import os
sys.path.insert(0, 'core/')
import one_D_scattering as ods
from wave_pack import convolution, delay
from wave_pack import gauss_pulse as pulse
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import cumtrapz as ct
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.optimize import bisect

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


def extract_eit_window(freq, transmittance, do=1e-5):

    """
    :param freq: set of frequencies
    :param transmittance: transmittance as complex function of freq
    :param do: \delta\omega to calculate derivative with
    :return: Transparency window and reversed delay time.

    We should consider pulse width within the borders:

    Sp. size < Pulse width < Transparency window

    A simple theory gives Sp. size \propto N and T. window \propto \sqrt{N}. We'll see.
    """

    tfunc = interp1d(freq, abs(transmittance)**2, kind='cubic')
    tfunc1 = lambda x: tfunc(x) - (1 - 1/np.e)
    domega = bisect(tfunc1, 0, 0.5*ods.RABI, xtol=0.001) # transparancy window
    imfunc = interp1d(freq, np.imag(transmittance), kind = 'cubic')
    spsize =  (imfunc(do) - imfunc(-do)) / (2*do)  #1 / delay time


    return 1/domega, spsize

CSVPATH = 'data/m_arrays/m_eit/'
if not os.path.exists(CSVPATH):
    os.makedirs(CSVPATH)


def toMathematica(filename, *argv):
    toCsv = np.column_stack(argv)
    np.savetxt(CSVPATH + filename + '.csv', toCsv, delimiter=',', fmt='%1.8f')

def save_timerange(args, filename):
    nrange = np.logspace(1, 3, 20, dtype=np.int)
    spsize = np.empty_like(nrange, dtype=np.float)
    windsize = np.empty_like(nrange, dtype=np.float)
    raz = 400
    freq = np.linspace(-0.1, 2 * ods.RABI + 0.1, raz)
    args['deltaP'] = freq

    for i, n in enumerate(nrange):
        args['nat'] = n
        print(f'calculation a chain of {n} atoms')
        elasticSc = ods.ensemble(**args)
        elasticSc.generate_ensemble()

        windsize[i], spsize[i] = extract_eit_window(freq, elasticSc.Transmittance)

    toMathematica(filename, nrange, windsize, spsize)

def save_pulse_scattering(args, filename):
    raz = 2000
    freq = np.linspace(-1.1, 1.1, raz)
    args['deltaP'] = freq
    elasticSc = ods.ensemble(**args)
    elasticSc.generate_ensemble()
    _, ft = convolution(freq, elasticSc.Transmittance, pulse(freq + SHIFT, pdTime), vg=0.7)
    _, fr = convolution(freq, elasticSc.Reflection, pulse(freq + SHIFT, pdTime), vg=0.7)
    t, fin = convolution(freq, np.ones_like(freq), pulse(freq + SHIFT, pdTime), vg=0.7)

    toMathematica(filename, t, abs(fin)**2, abs(ft)**2, abs(fr)**2)

ods.PAIRS = False
ods.RABI = 1
#ods.SINGLE_RAMAN = False
ods.DC = 0#-2#6#7
SHIFT = 0#2.42#5.04#-0.8#3.56#4.65#4.65#2.80#ods.DC-10.37
raz = 4000
freq = np.linspace(-1.1,1.1, raz)
#Pulse duration
pdTime = 1/70#26.7#26.9#2*np.pi
vis = True
noa = 1000
sq_reduce = lambda A: np.add.reduce(np.square(np.absolute(A)), axis=1)
ods.RABI_HYP = False



args = {
    'nat': noa,  # number of atoms
    'nb': 0,  # number of neighbours in raman chanel (for L-atom only)
    's': 'chain',  # Stands for atom positioning : chain, nocorrchain and doublechain
    'dist': 0.,  # sigma for displacement (choose 'chain' for gauss displacement.)
    'd': 1.5,  # distance from fiber
    'l0': 2.002/2,  # mean distance between atoms (in lambda_m /2 units)
    'deltaP': freq,  # array of freq.
    'typ': 'L',  # L or V for Lambda and V atom resp.
    'ff': 0.3
    }

raz = 400
freq = np.linspace(-0.1, 2*ods.RABI+0.1, raz)

save_timerange(args, 'timeDl')
save_pulse_scattering(args, 'timeDl_p')

args['l0'] = 2./2
save_timerange(args, 'timeBragg')
save_pulse_scattering(args, 'timeBragg_p')

args['s'] = 'nocorrchain'
save_timerange(args, 'timeNc')
save_pulse_scattering(args, 'timeNc_p')

args['s'] = 'chain'
args['l0'] = 3./2
save_timerange(args, 'timeTq')
save_pulse_scattering(args, 'timeTq_p')
