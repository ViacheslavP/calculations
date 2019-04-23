import sys
import os
sys.path.insert(0, 'core/')
import one_D_scattering as ods
from wave_pack import convolution, delay
from wave_pack import gauss_pulse_v2 as pulse
import numpy as np
from matplotlib import pyplot as plt
sq_reduce = lambda A: np.add.reduce(np.square(np.absolute(A)), axis=1)

CSVPATH = 'data/m_arrays/m_rabiqsn/'
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
ods.RABI = 1
ods.DC = 0# Control field detuning
SHIFT = 0# Carrier frequency of a photon
freq = np.linspace(-2.5,2.5, 980)
#Pulse duration
pdTime = 2*np.pi # pulse time, gammma^-1

noa = 200 #Number of atoms

ods.RABI = np.sqrt(noa/(6*np.pi)/(2*2*np.pi))


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

nats = np.linspace(100,1000,10,dtype=np.int)
#Bragg:

toMathematica('bragg', nats, *returnEnT(nats, args))
#With delta lambda:
args['l0'] = 1.001
toMathematica('dl', nats, *returnEnT(nats, args))

#Uncorrelated:
args['s'] = 'nocorrchain'
toMathematica('unc', nats, *returnEnT(nats, args))