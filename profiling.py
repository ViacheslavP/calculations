#!/usr/bin/env python
# encoding: utf-8
# filename: profile.py

import pstats, cProfile
import numpy as np

from one_D_scattering import ensemble
def demo_run():
    args = {

        'nat': 600,  # number of atoms
        'nb': 0,  # number of neighbours in raman chanel (for L-atom only)
        's': 'chain',  # Stands for atom positioning : chain, nocorrchain and doublechain
        'dist': 0,  # sigma for displacement (choose 'chain' for gauss displacement.)
        'd': 2.0,  # distance from fiber
        'l0': 1.0025,  # mean distance between atoms (in lambda_m /2 units)
        'deltaP': np.linspace(-10, 10, 180),  # array of freq.
        'typ': 'V',  # L or V for Lambda and V atom resp.
        'ff': 0.3

    }

    chi = ensemble(**args)
    chi.generate_ensemble()

cProfile.runctx("demo_run()", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
