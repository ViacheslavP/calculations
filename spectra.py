import sys
import os
import core.one_D_scattering as ods
from core.wave_pack import convolution, delay
from core.wave_pack import gauss_pulse_v2 as pulse
import numpy as np
sq_reduce = lambda A: np.add.reduce(np.square(np.absolute(A)), axis=1)

try:
    assert(type(sys.argv[1]) is str)
except AssertionError:
    print('Something went wrong...')
except IndexError:
    print('Using default CSVPATH (NOT FOR SERVER)')
    CSVPATH = 'data/m_arrays/m_spectra/'
else:
    CSVPATH = '/shared/data_m/' + sys.argv[1] + '/'

if not os.path.exists(CSVPATH):
    os.makedirs(CSVPATH)

def toMathematica(filename, *argv):
    toCsv = np.column_stack(argv)
    np.savetxt(CSVPATH + filename + '.csv', toCsv, delimiter=',', fmt='%1.8f')

def returnEnT(nats, args):
    cb = np.empty_like(nats, dtype=np.float)
    cb_t = np.empty_like(nats, dtype=np.float)
    for i,nat in enumerate(nats):
        args['nat'] = nat
        ods.RABI = np.sqrt(nat / (6 * np.pi) / (2 * 2 * np.pi))
        print(ods.RABI)
        #calculate spectra
        elasticSc = ods.ensemble(**args)
        elasticSc.generate_ensemble()

        #Convolve pulse fourier transformation with transmittance, reflection and 1 (to obtain initial temporal profile)
        _, ft = convolution(freq, elasticSc.Transmittance, pulse(freq+SHIFT, pdTime))
        _, fr = convolution(freq, elasticSc.Reflection, pulse(freq+SHIFT, pdTime))
        ti, fi = convolution(freq, np.ones_like(freq), pulse(freq+SHIFT, pdTime))

        cb[i] = sum(abs(ft)**2) / sum(abs(fi)**2)
        cb_t[i] = sum(ti*abs(ft)**2) / np.sqrt(sum(ti**2 *abs(fi)**2)) / cb[i]

    return cb, cb_t



"""
Control field and calculation parameters:
"""
ods.RABI_HYP = False
ods.PAIRS = False
ods.RABI = 0
ods.DC = 0# Control field detuning
SHIFT = 0# Carrier frequency of a photon
freq = np.linspace(-20,20,980)
#Pulse duration
pdTime = 2*np.pi # pulse time, gammma^-1

noa = 1000 #Number of atoms



args = {
    'nat': noa,  # number of atoms
    'nb': 0,  # number of neighbours in raman chanel (for L-atom only)
    's': 'chain',  # Stands for atom positioning : chain, nocorrchain and doublechain
    'dist': 0.,  # sigma for displacement (choose 'chain' for gauss displacement.)
    'd': 1.5,  # distance from fiber
    'l0': 2.01/2,  # mean distance between atoms (in lambda_m /2 units)
    'deltaP': freq,  # array of freq.
    'typ': 'L',  # L or V for Lambda and V atom resp.
    'ff': 0.3
    }
ods.RADIATION_MODES_MODEL = 1
se0 = ods.ensemble(**args)
se0.generate_ensemble()

toMathematica('withDDI', freq, se0.fullTransmittance, se0.fullReflection)

ods.RADIATION_MODES_MODEL = 0
se0 = ods.ensemble(**args)
se0.generate_ensemble()

toMathematica('withoutDDI', freq, se0.fullTransmittance, se0.fullReflection)