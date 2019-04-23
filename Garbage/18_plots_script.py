import sys
sys.path.insert(0, 'core/')
import one_D_scattering as ods
import numpy as np
import matplotlib.pyplot as plt

import os, errno


"""
That is what script calculate:

numberOfAtoms is obviously number of atoms - it is default [100,500,1000]  - will calculate 3 chains consisting 
of 100, 500, and 1000 atoms

ddInteraction could be on, off, or both (default)  
"""
numberOfAtoms = [100, 500, 1000]
ddInteraction = ['on', 'off']
separator = '`,' #separator of txt outputs

plots_data ='data'
try:
    os.makedirs(plots_data)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

plots_arrays = 'data/m_arrays'
try:
    os.makedirs(plots_arrays)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
ods.RABI = 0.


freq = np.linspace(-50.5,50.5, 800)
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



args_commensurate = args.copy()
args_disorder = args.copy()
args_noncommensurate = args.copy()
args_lambda = args.copy()

args_noncommensurate['l0'] = 2.002/2. #lambda / 2 + 0.01 * lambda / 2
args_disorder['s'] = 'nocorrchain'
args_lambda['l0'] = 2.


ods.RABI = 0.
for noa in numberOfAtoms:
    for switch in ddInteraction:
        if switch=='on':
            ods.RADIATION_MODES_MODEL = 1.
        elif switch=='off':
            ods.RADIATION_MODES_MODEL = 0
        else:
            raise NameError('Something went wrong!')

        args_commensurate['nat'] = noa
        args_disorder['nat'] = noa
        args_noncommensurate['nat'] = noa
        args_lambda['nat'] = noa

        ens_commensurate = ods.ensemble(**args_commensurate)
        ens_noncommensurate = ods.ensemble(**args_noncommensurate)
        ens_disorder = ods.ensemble(**args_disorder)
        ens_lambda = ods.ensemble(**args_lambda)

        ens_commensurate.generate_ensemble()
        ens_disorder.generate_ensemble()
        ens_noncommensurate.generate_ensemble()
        ens_lambda.generate_ensemble()

        #Rayleigh+Raman

        rmnTransCommensurate = ens_commensurate.fullTransmittance
        rmnReflCommensurate = ens_commensurate.fullReflection

        rmnTransNoncommensurate = ens_noncommensurate.fullTransmittance
        rmnReflNoncommensurate = ens_noncommensurate.fullReflection

        rmnTransDisorder = ens_disorder.fullTransmittance
        rmnReflDisorder = ens_disorder.fullReflection

        rmnTransLambda = ens_lambda.fullTransmittance
        rmnReflLambda = ens_lambda.fullReflection
        #Rayleigh

        rlghTransCommensurate = abs(ens_commensurate.Transmittance)**2
        rlghReflCommensurate = abs(ens_commensurate.Reflection)**2

        rlghTransNoncommensurate = abs(ens_noncommensurate.Transmittance)**2
        rlghReflNoncommensurate = abs(ens_noncommensurate.Reflection)**2

        rlghTransDisorder = abs(ens_disorder.Transmittance)**2
        rlghReflDisorder = abs(ens_disorder.Reflection)**2

        rlghTransLambda = abs(ens_lambda.Transmittance)**2
        rlghReflLambda = abs(ens_lambda.Reflection)**2





        np.savez(plots_data+'/NOA=%d' % noa+'%s.npz' % switch,
             frequencies = freq,
             rmnTransCommensurate = rmnTransCommensurate,
             rmnReflCommensurate = rmnReflCommensurate,
             rmnTransNoncommensurate = rmnTransNoncommensurate,
             rmnReflNoncommensurate = rmnReflNoncommensurate,
             rmnTransDisorder = rmnTransDisorder,
             rmnReflDisorder = rmnReflDisorder,
             rmnTransLambda = rmnTransLambda,
             rmnReflLambda = rmnReflLambda,
             rlghTransCommensurate = rlghTransCommensurate,
             rlghReflCommensurate = rlghReflCommensurate,
             rlghTransNoncommensurate = rlghTransNoncommensurate,
             rlghReflNoncommensurate = rlghReflNoncommensurate,
             rlghTransDisorder = rlghTransDisorder,
             rlghReflDisorder = rlghReflDisorder,
             rlghTransLambda = rlghTransLambda,
             rlghReflLambda = rlghReflLambda
        )





ods.RABI = 4.
for noa in numberOfAtoms:
    for switch in ddInteraction:
        if switch=='on':
            ods.RADIATION_MODES_MODEL = 1.
        elif switch=='off':
            ods.RADIATION_MODES_MODEL = 0
        else:
            raise NameError('Something went wrong!')

        args_commensurate['nat'] = noa
        args_disorder['nat'] = noa
        args_noncommensurate['nat'] = noa
        args_lambda['nat'] = noa



        ens_commensurate = ods.ensemble(**args_commensurate)
        ens_noncommensurate = ods.ensemble(**args_noncommensurate)
        ens_disorder = ods.ensemble(**args_disorder)
        ens_lambda = ods.ensemble(**args_lambda)

        ens_commensurate.generate_ensemble()
        ens_disorder.generate_ensemble()
        ens_noncommensurate.generate_ensemble()
        ens_lambda.generate_ensemble()

        #Rayleigh+Raman

        rmnTransCommensurate = ens_commensurate.fullTransmittance
        rmnReflCommensurate = ens_commensurate.fullReflection

        rmnTransNoncommensurate = ens_noncommensurate.fullTransmittance
        rmnReflNoncommensurate = ens_noncommensurate.fullReflection

        rmnTransDisorder = ens_disorder.fullTransmittance
        rmnReflDisorder = ens_disorder.fullReflection

        rmnTransLambda = ens_lambda.fullTransmittance
        rmnReflLambda = ens_lambda.fullReflection
        #Rayleigh

        rlghTransCommensurate = abs(ens_commensurate.Transmittance)**2
        rlghReflCommensurate = abs(ens_commensurate.Reflection)**2

        rlghTransNoncommensurate = abs(ens_noncommensurate.Transmittance)**2
        rlghReflNoncommensurate = abs(ens_noncommensurate.Reflection)**2

        rlghTransDisorder = abs(ens_disorder.Transmittance)**2
        rlghReflDisorder = abs(ens_disorder.Reflection)**2

        rlghTransLambda = abs(ens_lambda.Transmittance)**2
        rlghReflLambda = abs(ens_lambda.Reflection)**2

        elasticTransCommensurate = ens_commensurate.Transmittance
        elasticTransNoncommensurate = ens_noncommensurate.Transmittance
        elasticTransDisorer = ens_disorder.Transmittance
        elasticTransLambda = ens_lambda.Transmittance

        np.savez(plots_data + '/EIT_NOA=%d' % noa + '%s.npz' % switch,
                 frequencies=freq,
                 rmnTransCommensurate=rmnTransCommensurate,
                 rmnReflCommensurate=rmnReflCommensurate,
                 rmnTransNoncommensurate=rmnTransNoncommensurate,
                 rmnReflNoncommensurate=rmnReflNoncommensurate,
                 rmnTransDisorder=rmnTransDisorder,
                 rmnReflDisorder=rmnReflDisorder,
                 rmnTransLambda=rmnTransLambda,
                 rmnReflLambda=rmnReflLambda,
                 rlghTransCommensurate=rlghTransCommensurate,
                 rlghReflCommensurate=rlghReflCommensurate,
                 rlghTransNoncommensurate=rlghTransNoncommensurate,
                 rlghReflNoncommensurate=rlghReflNoncommensurate,
                 rlghTransDisorder=rlghTransDisorder,
                 rlghReflDisorder=rlghReflDisorder,
                 rlghTransLambda=rlghTransLambda,
                 rlghReflLambda=rlghReflLambda,
                 elasticTransCommensurate = elasticTransCommensurate,
                 elasticTransDisorer = elasticTransDisorer,
                 elasticTransLambda = elasticTransLambda,
                 elasticTransNoncommensurate = elasticTransNoncommensurate
                 )
np.savez(plots_data+'/meta.npz', numberOfAtoms = numberOfAtoms, ddInteraction = ddInteraction)
print('DONE!')

