"""

Script is supposed to do calculations on server 

directory must include:

one_D_scattering.py
bmode.py

"""

import numpy as np
from one_D_scattering import ensemble

from time import time

 
args = {
        'nat':100, #number of atoms
        'nb':0, #number of neighbours in raman chanel (for L-atom only)
        's':'ff_chain', #Stands for atom positioning : chain, nocorrchain and doublechain
        'dist':0,  # sigma for displacement (choose 'chain' for gauss displacement.)
        'd' : 2.0, # distance from fiber (in units of fiber's radius
        'l0':1.0025, # mean distance between atoms (lambdabar units)
        'deltaP':np.linspace(-10, 10, 250),  # array of freq.
        'typ':'V',  # L or V for Lambda and V atom resp.
        'ff': 0.3
        }


chi = ensemble(**args)
chi.generate_ensemble()

np.savez('arrays.npz', T=chi.Transmittance, R=chi.Reflection, iT=chi.iTransmittance, iR=chi.iReflection)

