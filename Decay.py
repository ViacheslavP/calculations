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
c = 1
kv =1 
d00 = 0.5

args = {'k': 1, 
        'n': 1.45,
        'a': a
        }
m = exact_mode(**args)


x = np.linspace(1, 3, ndots) * a

em = -m.E(x)
ez = m.Ez(x)
ep = -m.dE(x)


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
    
    

    



"""
___________________________________________________

V - atom
___________________________________________________
"""


Sigma = np.zeros([3,3,ndots],dtype=complex)

for i in range(ndots):
    
    
    emfi  =                np.array([ep[i], ez[i], em[i]], dtype=complex)
    emfic =  np.conjugate( np.array([ep[i], ez[i], em[i]], dtype=complex) )
                    
    epfi  =  np.conjugate( np.array([em[i], ez[i], ep[i]], dtype=complex) )
    epfic =                np.array([em[i], ez[i], ep[i]], dtype=complex)
                    
                    
                    
    embi  =                np.array([-ep[i], ez[i], -em[i]], dtype=complex)
    embic =  np.conjugate( np.array([-ep[i], ez[i], -em[i]], dtype=complex) )
                    
    epbi  =  np.conjugate( np.array([-em[i], ez[i], -ep[i]], dtype=complex) )
    epbic =                np.array([-em[i], ez[i], -ep[i]], dtype=complex)
    
    
                    
    forward = np.outer(emfi, emfic) + np.outer(epfi, epfic)
    backward = np.outer(embi, embic) + np.outer(epbi, epbic)
    
    
    Sigma[:,:,i] = 0.5*d00*d00 * 3 *(forward+backward)*-1*2j*np.pi*kv*(1/m.vg - 1/c) 
    

Sigma[1,1,:] += Sigma[1,1,:]* (1/c)/(1/m.vg - 1/c)  

plt.plot(x/a,(1+2*gamma), 'g-', lw=1.5, label = '$\gamma_{\Lambda}$')   
plt.plot(x/a,(1+2j*Sigma[0,0,:]), 'r-', lw=1.5, label = '$\gamma_{++}$')
plt.plot(x/a,(1+2j*Sigma[1,1,:]), 'b-', lw=1.5, label = '$\gamma_{00}$')    

plt.ylabel('$\gamma / \gamma_0$')
plt.xlabel('$r / a$')
plt.legend()   
    
plt.savefig('gammas.svg', dpi = 600)
plt.show()    



