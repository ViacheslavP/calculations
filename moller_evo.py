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
ods.RESONATOR = 0
dicke = ods.ensemble(**args)
dicke.generate_ensemble()

ods.RESONATOR = 1
resonator = ods.ensemble(**args)
resonator.generate_ensemble()


dAD = np.zeros_like(dicke.AtomicDecay)
rAD = np.zeros_like(dicke.AtomicDecay)

for i in range(args['nat']):
    t, dAD[i, :] = convolution(freq, dicke.AtomicDecay[i, :], np.ones_like(freq))
    _, rAD[i, :] = convolution(freq, resonator.AtomicDecay[i, :], np.ones_like(freq))

dickeDecay = np.dot(np.conj(np.transpose(dAD)), dAD).diagonal()
resDecay = np.dot(np.conj(np.transpose(rAD)), rAD).diagonal()


plt.xlim([-2,2])
plt.plot(t, dickeDecay / dickeDecay.max(), 'r')
plt.plot(t, resDecay / resDecay.max(), 'g')
plt.show()

#Second figure
from scipy.optimize import curve_fit

def extract_time(t, et):
    def _func(x, g, A):
        return A*np.exp(-1*x*g) #np.A*np.exp(-x*g)

    lent = len(t)
    lens = np.int(lent//1.8)
    t = t[lens:lent]
    et = et[lens:lent]

    #plt.plot(t, et)
    #plt.plot(t, et - np.amax(et)*_func(t,1,1))
    #plt.show()

    popt, pcov = curve_fit(_func, t, et/np.amax(et), bounds=(0, 3), method='trf')

    return popt[0]

ods.RESONATOR = 0
#args['nat'] = 100
phases = np.linspace(0, 7*np.pi/8, np.pi)
gmss = np.empty_like(phases)

for k, phase in enumerate(phases):
    ods.PHASE = phase

    resonator = ods.ensemble(**args)
    resonator.generate_ensemble()
    for i in range(args['nat']):
        t, rAD[i, :] = convolution(freq, resonator.AtomicDecay[i, :], np.ones_like(freq))
        #rAD[i, :] *= np.exp(1j*i*phase)

    resDecay = np.dot(np.conj(np.transpose(rAD)), rAD).diagonal()
    gmss[k] = extract_time(t, resDecay)

plt.ylim(0, 1.5)
plt.plot(phases / 0.5 / np.pi, gmss)
plt.show()

colors = ['r', 'g', 'k']
phases = [np.pi]
args['nat'] = 100
for k, phase in enumerate(phases):
    ods.PHASE = phase

    resonator = ods.ensemble(**args)
    resonator.generate_ensemble()
    for i in range(args['nat']):
        t, rAD[i, :] = convolution(freq, resonator.AtomicDecay[i, :], np.ones_like(freq))
        #rAD[i, :] *= np.exp(1j*i*phase)

    resDecay = np.dot(np.conj(np.transpose(rAD)), rAD).diagonal()

    plt.plot(t, np.log(resDecay/np.amax(resDecay)), colors[k])

plt.ylim(-3,0)
plt.xlim(0, 2)
plt.show()

logresD_0 = np.log(resDecay/np.amax(resDecay))

ods.RESONATOR = 1
args['nat'] = 300
colors = ['r', 'g', 'k']
for k, phase in enumerate(phases):
    ods.PHASE = phase

    resonator = ods.ensemble(**args)
    resonator.generate_ensemble()
    for i in range(args['nat']):
        t, rAD[i, :] = convolution(freq, resonator.AtomicDecay[i, :], np.ones_like(freq))
        #rAD[i, :] *= np.exp(1j*i*phase)

    resDecay = np.dot(np.conj(np.transpose(rAD)), rAD).diagonal()

    plt.plot(t, np.log(resDecay/np.amax(resDecay)), colors[k])

logresD_1 = np.log(resDecay/np.amax(resDecay))
plt.ylim(-3,0)
plt.xlim(0, 2)
plt.show()

ods.RESONATOR = 2
args['nat'] = 200
colors = ['r', 'g', 'k']
for k, phase in enumerate(phases):
    ods.PHASE = phase

    resonator = ods.ensemble(**args)
    resonator.generate_ensemble()
    for i in range(args['nat']):
        t, rAD[i, :] = convolution(freq, resonator.AtomicDecay[i, :], np.ones_like(freq))
        #rAD[i, :] *= np.exp(1j*i*phase)

    resDecay = np.dot(np.conj(np.transpose(rAD)), rAD).diagonal()

    plt.plot(t, np.log(resDecay/np.amax(resDecay)), colors[k])
logresD_2 = np.log(resDecay/np.amax(resDecay))
plt.ylim(-3,0)
plt.xlim(0, 2)
plt.show()


ods.RESONATOR = 3
args['nat'] = 800
colors = ['r', 'g', 'k']
for k, phase in enumerate(phases):
    ods.PHASE = phase

    resonator = ods.ensemble(**args)
    resonator.generate_ensemble()
    rAD = np.zeros_like(resonator.AtomicDecay)
    for i in range(args['nat']):
        t, rAD[i, :] = convolution(freq, resonator.AtomicDecay[i, :], np.ones_like(freq))
        #rAD[i, :] *= np.exp(1j*i*phase)

    resDecay = np.dot(np.conj(np.transpose(rAD)), rAD).diagonal()

    plt.plot(t, np.log(resDecay/np.amax(resDecay)), colors[k])

logresD_3 = np.log(resDecay/np.amax(resDecay))
plt.ylim(-3,0)
plt.xlim(0, 2)
plt.show()

ods.RESONATOR = 4
args['nat'] = 900
for k, phase in enumerate(phases):
    ods.PHASE = phase

    resonator = ods.ensemble(**args)
    resonator.generate_ensemble()
    rAD = np.zeros_like(resonator.AtomicDecay)
    for i in range(args['nat']):
        t, rAD[i, :] = convolution(freq, resonator.AtomicDecay[i, :], np.ones_like(freq))
        #rAD[i, :] *= np.exp(1j*i*phase)

    resDecay = np.dot(np.conj(np.transpose(rAD)), rAD).diagonal()

    plt.plot(t, np.log(resDecay/np.amax(resDecay)), colors[k])

logresD_4 = np.log(resDecay/np.amax(resDecay))
plt.ylim(-3,0)
plt.xlim(0, 2)
plt.show()


plt.plot(t, logresD_0, 'g')
plt.plot(t, logresD_1, 'm-.')
plt.plot(t, logresD_2, 'b--')
plt.plot(t, logresD_3, 'k')
plt.plot(t, logresD_3, 'r-.')
plt.ylim(-3,0)
plt.xlim(0, 2)
plt.show()

