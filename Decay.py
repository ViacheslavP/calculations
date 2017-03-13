"""

@author: ViacheslavP

Script for drawing of decay rate of Lambda-atom and for V-atom

"""
import numpy as np
import matplotlib.pyplot as plt
from bmode import exact_mode

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size':16})
rc('text', usetex=True)


a = 2*np.pi*200/780
ndots = 100
c = 100000000000000000000
kv =1 
d00 = 0.5

args = {'k': 1, 
        'n': 1.452,
        'a': a
        }
m = exact_mode(**args)


x = np.linspace(1, 3, ndots) * a

em = -m.E(x)
ez = m.Ez(x)
ep = m.dE(x)


"""
___________________________________________________

Lambda - atom
___________________________________________________
"""
gamma = np.zeros([ndots],dtype=complex)

for i in range(ndots):
    emfi  =                np.array([ep[i], ez[i], em[i]], dtype=complex)
    emfic =  np.conjugate( np.array([ep[i], ez[i], em[i]], dtype=complex) )
                    
    gamma[i] = 2*d00*d00 *( (np.dot(emfi,emfic)* 2*np.pi*kv*(1/m.vg - 1/c)  + ez[i]*ez[i]*2*np.pi*kv/c))
    
    

    print(2*gamma)







