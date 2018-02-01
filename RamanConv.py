import sys
sys.path.insert(0, 'core/')
import one_D_scattering as ods
import numpy as np
import matplotlib.pyplot as plt

#Consider chain with 50 atoms placed evenly with its distance lambda/2
ods.RABI = 0.

freq = np.linspace(-2.5,2.5, 480)
args = {

    'nat': 50,  # number of atoms
    'nb': 0,  # number of neighbours in raman chanel (for L-atom only)
    's': 'chain',  # Stands for atom positioning : chain, nocorrchain and doublechain
    'dist': 0.,  # sigma for displacement (choose 'chain' for gauss displacement.)
    'd': 1.5,  # distance from the fiber
    'l0': 1.0/2,  # mean distance between atoms (in lambda_m /2 units)
    'deltaP': freq,  # array of freq.
    'typ': 'L',  # L or V for Lambda and V atom resp.
    'ff': 0.3
    }


"""
VAR is the variant of meta-calculation
1. Accuracy of our approximation
2. Reflection and Transmission saturation
3. Constructive and destructive interference
4. ...
"""
VAR = 3
if VAR == 1:

    #First of all, lets find a role of neighbours
    args['nb'] = 4
    ens_four = ods.ensemble(**args)
    ens_four.generate_ensemble()

    args['nb'] = 2
    ens_two = ods.ensemble(**args)
    ens_two.generate_ensemble()

    args['nb'] = 0
    ens_zero = ods.ensemble(**args)
    ens_zero.generate_ensemble()


    plt.plot(freq, ens_two.fullTransmittance-ens_zero.fullTransmittance)
    plt.plot(freq, ens_four.fullTransmittance-ens_zero.fullTransmittance)
    plt.show()

    plt.plot(freq, ens_two.fullReflection-ens_zero.fullReflection)
    plt.plot(freq, ens_four.fullReflection-ens_zero.fullReflection)
    plt.show()

    plt.plot(freq, ens_two.fullTransmittance)
    plt.plot(freq, abs(ens_two.Transmittance)**2)
    plt.show()

    plt.plot(freq, ens_two.fullReflection)
    plt.plot(freq, abs(ens_two.Reflection)**2)
    plt.show()

elif VAR == 2:

    ref = []
    trans = []
    ref_two = []
    trans_two = []
    maxAtoms = 80
    numberOfAtoms = range(60, maxAtoms+5, 5)
    for noa in numberOfAtoms:
        args['nb'] = 0
        args['nat'] = noa
        ens = ods.ensemble(**args)
        ens.generate_ensemble()
        ref.append(np.amax((ens.fullReflection)))
        trans.append(np.amin((ens.fullTransmittance)))

        args['nb'] = 2
        ens = ods.ensemble(**args)
        ens.generate_ensemble()
        ref_two.append(np.amax((ens.fullReflection)))
        trans_two.append(np.amin((ens.fullTransmittance)))
        print(noa, '/%d atoms done' % maxAtoms)


    plt.plot(numberOfAtoms, ref, 'b-', label='Maximum of Reflection')
    plt.plot(numberOfAtoms, trans, 'r-', label='Minimum of Transmission')
    plt.plot(numberOfAtoms, ref_two, 'b--', label='Maximum of Reflection (two-atom cluster)')
    plt.plot(numberOfAtoms, trans_two, 'r--', label='Minimum of Transmission (two-atom cluster)')
    plt.xlabel('Number of atoms')
    plt.legend()
    plt.show()

elif VAR == 3:

    innerDistance = np.linspace(0.5, 1.5, 120)
    ref = []
    ref_el = []
    trans = []
    trans_el = []
    for dist in innerDistance:
        args['l0'] = dist

        ens = ods.ensemble(**args)
        ens.generate_ensemble()
        ref.append(np.amax((ens.fullReflection)))
        trans.append(np.amin((ens.fullTransmittance)))
        ref_el.append(np.amax(abs(ens.Reflection) ** 2))
        trans_el.append(np.amin(abs(ens.Transmittance) ** 2))

    plt.plot(innerDistance, ref, 'm-', label=r'Total Maximum Reflection')
    plt.plot(innerDistance, ref_el, 'm--', label=r'Elastic Rayleigh Maximum Reflection')
    plt.legend()
    plt.show()

    plt.plot(innerDistance, trans, 'k-', label='Total Maximum Transmission')
    plt.plot(innerDistance, trans_el, 'k--', label='Elastic Maximum Transmission')
    plt.legend()
    plt.show()