import sys, os
sys.path.insert(0, 'core/')
import one_D_scattering as ods
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import cumtrapz as ct
from wave_pack import convolution, delay
from wave_pack import inverse_pulse as pulse
from wave_pack import efficiency



ods.PAIRS = True

#Control mode Setup
ods.RABI = 0
ods.DC = -4#6#7

#Pulse setup
pdGamma = 0.05#26.7#26.9#2*np.pi
pdShift = +4

#Number of atoms
noa = 300


#Calculation Scheme Setup
ods.RABI_HYP = False
ods.SINGLE_RAMAN = True
ods.RADIATION_MODES_MODEL = 1
ods.VACUUM_DECAY = 1
ods.DDI = 1
raz = 1200
freq = np.linspace(-380.5,380.5, raz)
CSVPATH = 'data/m_arrays/m_pics_23.02.2019/'


if not os.path.exists(CSVPATH):
    os.makedirs(CSVPATH)

#First figure

args = {
    'nat': noa,  # number of atoms
    'nb': 0,  # number of neighbours in raman chanel (for L-atom only)
    's': 'chain',  # Stands for atom positioning : chain, nocorrchain and doublechain
    'dist': 0.,  # sigma for displacement (choose 'chain' for gauss displacement., \lambda/2 units)
    'd': 1.5,  # distance from fiber
    'l0': 2. / 2,  # mean distance between atoms (in lambda_m /2 units)
    'deltaP': freq,  # array of freq
    'typ': 'LM',  # L or V for Lambda and V atom resp.
    'ff': 0.3,  # filling factor (for ff_chain only)
    'numexc': 0
}
ods.PAIRS = False
ods.PHASE = np.pi
dicke = ods.ensemble(**args)
dicke.generate_ensemble()

args['s'] = 'resonator'
resonator = ods.ensemble(**args)
resonator.generate_ensemble()

ods.PHASE = 0
args['s'] = 'chain'
dickeDphsd = ods.ensemble(**args)
#dickeDphsd.generate_ensemble()


dAD = np.zeros_like(dicke.AtomicDecay)
rAD = np.zeros_like(dicke.AtomicDecay)

for i in range(args['nat']):
    t, dAD[i, :] = convolution(freq, dicke.AtomicDecay[i, :], np.ones_like(freq))
    _, rAD[i, :] = convolution(freq, resonator.AtomicDecay[i, :], np.ones_like(freq))

dickeDecay = np.dot(np.conj(np.transpose(dAD)), dAD).diagonal()
resDecay = np.dot(np.conj(np.transpose(rAD)), rAD).diagonal()


plt.xlim([0,4])
plt.ylim()
plt.yscale('log')
dcay = np.exp(-t)
plt.plot(t, dickeDecay / dickeDecay.max(), 'r')
plt.plot(t, resDecay / resDecay.max(), 'g')
plt.plot(t, dcay, 'k--')
plt.show()
