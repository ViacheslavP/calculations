import sys
import os
import core.one_D_scattering as ods
import numpy as np
from core.wave_pack import convolution, delay
from core.wave_pack import gauss_pulse_v2 as pulse
from core.wave_pack import efficiency_map, unitaryTransform, efficiency_sq
sq_reduce = lambda A: np.add.reduce(np.square(np.absolute(A)), axis=0)

try:
    assert(type(sys.argv[1]) is str)
except AssertionError:
    print('Something went wrong...')
except IndexError:
    print('Using default CSVPATH (NOT FOR SERVER)')
    CSVPATH = 'data/m_arrays/wpairs_nocorr/'
else:
    CSVPATH = '/shared/data_m/' + sys.argv[1] + '/'

if not os.path.exists(CSVPATH):
    os.makedirs(CSVPATH)

def toMathematica(filename, *argv):
    toCsv = np.column_stack(argv)
    np.savetxt(CSVPATH + filename + '.csv', toCsv, delimiter=',', fmt='%1.8f')

def saveEnsemble(filename, ensemble):
    np.savez(CSVPATH + filename,
             freq=freq,
             fullTransmittance=ensemble.fullTransmittance,
             fullReflection=ensemble.fullReflection,
             Transmittance=ensemble.Transmittance,
             Reflection=ensemble.Reflection,
             iTransmittance=ensemble.iTransmittance,
             iReflection=ensemble.iReflection,
             TF2_0m=ensemble.TF2_0m,
             TB2_0m=ensemble.TB2_0m,
             TF2_0p=ensemble.TF2_0p,
             TB2_0p=ensemble.TB2_0p,
             TF2_mm=ensemble.TF2_mm,
             TB2_mm=ensemble.TB2_mm,
             TF2_mp=ensemble.TF2_mp,
             TB2_mp=ensemble.TB2_mp)
    pass


#Calculation parameters

#Fix random numbers (every run of this program results in same atomic position if true)
ods.FIX_RANDOM = True
ods.SEED = 15

#Set frequency domain
raz = 980
freq = np.linspace(-25.5,25.5, raz)

#Use extended basis
#False - use neighbours (useless(?) for fiber), dim = 3^nb * N
#True - use pairs (resolvent operator with maximum one jump), dim = N + 2N(N-1) - Unstable for nocorrchain!
#It won't do any significant differance
ods.PAIRS = False


nat = 200
args = {
            'nat':nat, #number of atoms
            'nb':0, #number of neighbours in raman chanel (for L-atom only)
            's':'chain', #Stands for atom positioning : chain, nocorrchain and doublechain
            'dist':0.,  # sigma for displacement (choose 'chain' for gauss displacement., \lambda/2 units)
            'd' : 1.5, # distance from fiber
            'l0': 3.00/2, # mean distance between atoms (in lambda_m /2 units)
            'deltaP':freq,  # array of freq
            'typ':'L',  # L or V for Lambda and V atom resp.
            'ff': 0.3 #filling factor (for ff_chain only)
        }

#Set Control field
ods.RABI = np.sqrt(nat / (6 * np.pi) / (2 * 2 * np.pi))
ods.DC = 0
args['nat'] = nat
#Generate ensemble
shundred = ods.ensemble(**args)
shundred.generate_ensemble()
#saveEnsemble(shundred)


pdTime = 2*np.pi

#
tx = abs(shundred.Transmittance)**2
alignedTransmittance = np.piecewise(tx, [tx<-1,tx>1], )
shift = freq[np.argmax(alignedTransmittance)]

def opt_conv(x):
    return convolution(freq, x, pulse(freq + shift, pdTime))


time, fi = opt_conv(np.ones_like(shundred.Transmittance))

# Forward Intensity

#Elastic intensity, atoms and polarization remain in input state
_, ft = opt_conv(shundred.Transmittance)

#Polarization switches, atoms remain in +1 state
_, ft_i = opt_conv(shundred.iTransmittance)

#Polarization remains, 1 atom in 0-state
tf0m = np.empty((nat, len(ft)), dtype=np.complex)
#Polarization switches, 1 atom in 0-state
tf0p = np.empty((nat, len(ft)), dtype=np.complex)
#Polarization remains, atom in -1-state
tfmm = np.empty((nat, len(ft)), dtype=np.complex)
#Polarization switches, atom in -1 state (non-zero in free space)
tfmp = np.empty((nat, len(ft)), dtype=np.complex)

for i in range(nat):
    _, tf0m[i, :] = opt_conv(1j * shundred.TF2_0m[i, :])
    _, tf0p[i, :] = opt_conv(1j * shundred.TF2_0p[i, :])
    _, tfmm[i, :] = opt_conv(1j * shundred.TF2_mm[i, :])
    _, tfmp[i, :] = opt_conv(1j * shundred.TF2_mp[i, :])

#Collect full intensity
Forward = abs(ft)**2 \
        + abs(ft_i)**2 \
        + sq_reduce(tf0m) \
        + sq_reduce(tf0p) \
        + sq_reduce(tfmm) \
        + sq_reduce(tfmp)

import matplotlib.pyplot as plt
plt.plot(time, abs(fi)**2)
plt.plot(time, abs(ft)**2, 'r--')
plt.plot(time, Forward, 'r')
plt.xlim((-20, 60))
plt.show()