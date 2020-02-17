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
    CSVPATH = 'data/m_arrays/nocorr_09.08.2019/'
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
             SF1_pm=ensemble.SF1_pm,
             SB1_pm=ensemble.SB1_pm,
             SF1_pp=ensemble.SF1_pp,
             SB1_pp=ensemble.SB1_pp,
             SF2_0m=ensemble.SF2_0m,
             SB2_0m=ensemble.SB2_0m,
             SF2_0p=ensemble.SF2_0p,
             SB2_0p=ensemble.SB2_0p,
             SF2_mm=ensemble.SF2_mm,
             SB2_mm=ensemble.SB2_mm,
             SF2_mp=ensemble.SF2_mp,
             SB2_mp=ensemble.SB2_mp)
    pass

ods.FIX_RANDOM = True
ods.SEED = 12
raz = 8000
freq = np.linspace(-25.5,25.5, raz)
ods.PAIRS = False
ods.RABI = 2.
ods.DC = -4.

args = {
            'nat':10, #number of atoms
            'nb':0, #number of neighbours in raman chanel (for L-atom only)
            's':'nocorrchain', #Stands for atom positioning : chain, nocorrchain and doublechain
            'dist':0.,  # sigma for displacement (choose 'chain' for gauss displacement., \lambda/2 units)
            'd' : 1.5, # distance from fiber
            'l0': 2.00/2, # mean distance between atoms (in lambda_m /2 units)
            'deltaP':freq,  # array of freq
            'typ':'L',  # L or V for Lambda and V atom resp.
            'ff': 0.3 #filling factor (for ff_chain only)
        }


ods.OPPOSITE_SCATTERING = False

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


ods.OPPOSITE_SCATTERING = True

ods.RABI = 2.
ods.DC = -4.
args['nat'] = 10
sten = ods.ensemble(**args)
args['nat'] = 100
shundred = ods.ensemble(**args)

for filename, ensemble in (('invtenAT', sten), ('invhundredAT', shundred)):
    ensemble.generate_ensemble()
    saveEnsemble(filename, ensemble)


ods.RABI = 0.
ods.DC = -4.
args['nat'] = 10
sten = ods.ensemble(**args)
args['nat'] = 100
shundred = ods.ensemble(**args)

for filename, ensemble in (('invten', sten), ('invhundred', shundred)):
    ensemble.generate_ensemble()
    saveEnsemble(filename, ensemble)

