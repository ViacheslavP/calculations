import sys
sys.path.insert(0, 'core/')
import one_D_scattering as ods
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#Consider chain with 50 atoms placed evenly with its distance lambda/2
ods.RABI = 0.
ods.RAMAN_BACKSCATTERING = True

freq = np.linspace(-2.5,2.5, 180)
args = {

    'nat': 50,  # number of atoms
    'nb': 0,  # number of neighbours in raman chanel (for L-atom only)
    's': 'chain',  # Stands for atom positioning : chain, nocorrchain and doublechain
    'dist': 0.,  # sigma for displacement (choose 'chain' for gauss displacement.)
    'd': 2.0,  # distance from the fiber
    'l0': 2.0/2,  # mean distance between atoms (in lambda_m /2 units)
    'deltaP': freq,  # array of freq.
    'typ': 'L',  # L or V for Lambda and V atom resp.
    'ff': 0.3
    }

"""
The Idea is to estimate collectivity in such a system
"""
maxAtoms = 120
numberOfAtoms  = np.arange(5,maxAtoms,5, dtype=int)

map = np.empty([len(numberOfAtoms), maxAtoms, maxAtoms])
firstAtom = []
secondAtom = []
lastAtom = []

col_firstAtom = []
col_secondAtom = []
col_lastAtom = []

slight_firstAtom = []
slight_secondAtom = []
slight_lastAtom = []
for n in numberOfAtoms:
    args['nat'] = n
    args['l0'] = 0.5
    ens = ods.ensemble(**args)
    ens.generate_ensemble()
    firstAtom.append((np.amax(ens.RamanBackscattering, axis=0))[0])
    secondAtom.append((np.amax(ens.RamanBackscattering, axis=0))[1])
    lastAtom.append((np.amax(ens.RamanBackscattering, axis=0))[::-1][0])

    args['l0'] = 1.0
    ens = ods.ensemble(**args)
    ens.generate_ensemble()
    col_firstAtom.append((np.amax(ens.RamanBackscattering, axis=0))[0])
    col_secondAtom.append((np.amax(ens.RamanBackscattering, axis=0))[1])
    col_lastAtom.append((np.amax(ens.RamanBackscattering, axis=0))[::-1][0])

    args['l0'] = 1.002

    ens = ods.ensemble(**args)
    ens.generate_ensemble()
    slight_firstAtom.append((np.amax(ens.RamanBackscattering, axis=0))[0])
    slight_secondAtom.append((np.amax(ens.RamanBackscattering, axis=0))[1])
    slight_lastAtom.append((np.amax(ens.RamanBackscattering, axis=0))[::-1][0])


    print(n, '/%d atoms done' % maxAtoms)

plt.plot(numberOfAtoms, firstAtom, 'r-')
plt.plot(numberOfAtoms, lastAtom, 'r--')

plt.plot(numberOfAtoms, col_firstAtom, 'b-')
plt.plot(numberOfAtoms, col_lastAtom, 'b--')

plt.plot(numberOfAtoms, slight_firstAtom, 'y-')
plt.plot(numberOfAtoms, slight_lastAtom, 'y--')
plt.show()

