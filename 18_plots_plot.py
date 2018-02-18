import sys
sys.path.insert(0, 'core/')
import numpy as np
import matplotlib.pyplot as plt

import os, errno


plots_data ='data'
if not (os.path.exists(plots_data) and os.path.exists(plots_data+'/meta.npz')):
    raise NameError('You have to run 18_plots_script.py at first!')

plots = plots_data+'/plots'

try:
    os.makedirs(plots)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

meta = np.load(plots_data + '/meta.npz')
numberOfAtoms = meta['numberOfAtoms']
ddInteraction = meta['ddInteraction']

for noa in numberOfAtoms:
    for switch in ddInteraction:
        npzfile = np.load('data/NOA=%d' % noa + '%s.npz' % switch)
        freq = npzfile['frequencies']
        rmnTransCommensurate = npzfile['rmnTransCommensurate']
        rmnReflCommensurate = npzfile['rmnReflCommensurate']
        rmnTransNoncommensurate = npzfile['rmnTransNoncommensurate']
        rmnReflNoncommensurate = npzfile['rmnReflNoncommensurate']
        rmnTransDisorder = npzfile['rmnTransDisorder']
        rmnReflDisorder = npzfile['rmnReflDisorder']
        rlghTransCommensurate = npzfile['rlghTransCommensurate']
        rlghReflCommensurate = npzfile['rlghReflCommensurate']
        rlghTransNoncommensurate = npzfile['rlghTransNoncommensurate']
        rlghReflNoncommensurate = npzfile['rlghReflNoncommensurate']
        rlghTransDisorder = npzfile['rlghTransDisorder']
        rlghReflDisorder = npzfile['rlghReflDisorder']


        plt.title('N = %d' % noa + ' Commensurate. Dipole-dipole interaction is %s' % switch)
        plt.plot(freq, rmnTransCommensurate, 'r', label = 'Rayleigh + Raman transmission')
        plt.plot(freq, rmnReflCommensurate, 'b', label = 'Rayleigh + Raman reflection')
        plt.plot(freq, rlghTransCommensurate, 'r--', label = 'Rayleigh transmission')
        plt.plot(freq, rlghReflCommensurate, 'b--', label = 'Rayleigh reflection')
        plt.savefig(plots + '/NOA=%d' % noa + 'Commensurate, dd is %s.svg' %switch)
        plt.legend()
        plt.xlabel('Detuning, '+r'$\gamma$')
        plt.show()

        plt.title('N = %d'%noa + ' Noncommensurate. Dipole-dipole interaction is %s' % switch)
        plt.plot(freq, rmnTransNoncommensurate, 'r', label = 'Rayleigh + Raman transmission')
        plt.plot(freq, rmnReflNoncommensurate, 'b', label = 'Rayleigh + Raman reflection')
        plt.plot(freq, rlghTransNoncommensurate, 'r--', label = 'Rayleigh transmission')
        plt.plot(freq, rlghReflNoncommensurate, 'b--', label = 'Rayleigh reflection')
        plt.savefig(plots + '/NOA=%d' % noa + 'Noncommensurate, dd is %s.svg' %switch)
        plt.legend()
        plt.xlabel('Detuning, '+r'$\gamma$')
        plt.show()

        plt.title('N = %d'%noa + ' Disordered Transmission. Dipole-dipole interaction is %s' % switch)
        plt.plot(freq, rmnTransDisorder, 'r', label = 'Rayleigh + Raman transmission')
        plt.plot(freq, rlghTransDisorder, 'r--', label = 'Rayleigh transmission')
        plt.savefig(plots + '/NOA=%d' % noa + ' TransmittanceDisorder, dd is %s.svg' %switch)
        plt.legend()
        plt.xlabel('Detuning, '+r'$\gamma$')
        plt.show()

        plt.title('N = %d'%noa + 'Disordered Reflection. Dipole-dipole interaction is %s' % switch)
        plt.plot(freq, rmnReflDisorder, 'b', label = 'Rayleigh + Raman reflection')
        plt.plot(freq, rlghReflDisorder, 'b--', label = 'Rayleigh reflection')
        plt.savefig(plots + '/NOA=%d' % noa + 'ReflectionDisorder, dd is %s.svg' %switch)
        plt.legend()
        plt.xlabel('Detuning, '+r'$\gamma$')
        plt.show()

        plt.plot(freq, rmnReflNoncommensurate - rlghReflNoncommensurate)
        plt.show()