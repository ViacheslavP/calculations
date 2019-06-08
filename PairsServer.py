import sys
import os
import core.one_D_scattering as ods
import numpy as np
sq_reduce = lambda A: np.add.reduce(np.square(np.absolute(A)), axis=1)

try:
    assert(type(sys.argv[1]) is str)
except AssertionError:
    print('Something went wrong...')
except IndexError:
    print('Using default CSVPATH (NOT FOR SERVER)')
    CSVPATH = 'data/m_arrays/m_/'
else:
    CSVPATH = '/shared/data_m/' + sys.argv[1]

if not os.path.exists(CSVPATH):
    os.makedirs(CSVPATH)

def toMathematica(filename, *argv):
    toCsv = np.column_stack(argv)
    np.savetxt(CSVPATH + filename + '.csv', toCsv, delimiter=',', fmt='%1.8f')

def saveEnsemble(filename, ensemble):
    np.savez(filename,
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

freq = np.linspace(-15,15, 4980)
ods.PAIRS = True
args = {
            'nat':10, #number of atoms
            'nb':0, #number of neighbours in raman chanel (for L-atom only)
            's':'chain', #Stands for atom positioning : chain, nocorrchain and doublechain
            'dist':0.,  # sigma for displacement (choose 'chain' for gauss displacement., \lambda/2 units)
            'd' : 1.5, # distance from fiber
            'l0': 2.00/2, # mean distance between atoms (in lambda_m /2 units)
            'deltaP':freq,  # array of freq
            'typ':'L',  # L or V for Lambda and V atom resp.
            'ff': 0.3 #filling factor (for ff_chain only)
        }


ods.RABI = 2.
ods.DC = -4.
args['nat'] = 10
sten = ods.ensemble(**args)
args['nat'] = 100
shundred = ods.ensemble(**args)

for filename, ensemble in (('tenAT', sten), ('hundredAT', shundred)):
    ensemble.generate_ensemble()
    saveEnsemble(filename, ensemble)

ods.RABI = 0.
ods.DC = -4.
args['nat'] = 10
sten = ods.ensemble(**args)
args['nat'] = 100
shundred = ods.ensemble(**args)

for filename, ensemble in (('ten', sten), ('hundred', shundred)):
    ensemble.generate_ensemble()
    saveEnsemble(filename, ensemble)

ods['s'] = 'nocorrchain'
ods.RABI = 2.
ods.DC = -4.
args['nat'] = 10
sten = ods.ensemble(**args)
args['nat'] = 100
shundred = ods.ensemble(**args)

for filename, ensemble in (('tenATnc', sten), ('hundredATnc', shundred)):
    ensemble.generate_ensemble()
    saveEnsemble(filename, ensemble)

ods.RABI = 0.
ods.DC = -4.
args['nat'] = 10
sten = ods.ensemble(**args)
args['nat'] = 100
shundred = ods.ensemble(**args)

for filename, ensemble in (('tennc', sten), ('hundrednc', shundred)):
    ensemble.generate_ensemble()
    saveEnsemble(filename, ensemble)