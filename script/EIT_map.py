import numpy as np
import matplotlib.pyplot as plt

npzfile = np.load('../data/3delayFile.npz')

delayArray = npzfile['delay']
ampArray = npzfile['amplitude']
numberAtoms = npzfile['numbers']
distAtoms = npzfile['distances']
#approxDelayArray = npzfile['approxdelay']

plt.plot(distAtoms, delayArray[0,:], 'b', lw=2., label = 'EIT delay, '+r'$\gamma^{-1}$')
#plt.plot(distAtoms, approxDelayArray[0,:], 'm-')
#plt.plot(distAtoms, ampArray[0,:], 'r--', lw=2., label = 'Amplitude [arb.units]')
plt.xlabel('Distance between atoms, '+r'$\lambda / 2$')
plt.ylabel('Group delay of EIT, '+r'$\gamma^{-1}$')
#plt.xlim([0.8,1.2])
#plt.legend()
plt.show()