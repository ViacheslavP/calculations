"""

@author: ViacheslavP

Drawing of decay rate of Lambda-atom and for V-atom

"""
import numpy as np
import matplotlib.pyplot as plt
from bmode import exact_mode

a = 2*np.pi*200/850
ndots = 100
c = 1
kv =1 
d00 = 0.5

args = {'k': 1, 
        'n': 1.45,
        'a': a
        }
m = exact_mode(**args)


x = np.linspace(1, 5, ndots) * a

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
                    
    gamma[i] = d00*d00 *( (np.dot(emfi,emfic)* 2j*np.pi*kv*(1/m.vg - 1/c)  + ez[i]*ez[i]*2j*np.pi*kv/c))
    
    
                    

    
plt.plot(x/a,(1-2j*gamma), 'g-')   

plt.show() 



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
    
    metric_tensor = np.array([[0,0,-1],[0,1,0],[-1,0,0]], dtype=complex)
    Sigma[:,:,i] = np.dot( Sigma[:,:,i],metric_tensor) 
    

plt.plot(x/a,(1+2j*Sigma[0,0,:]), 'r-')
plt.plot(x/a,(1+2j*Sigma[1,1,:]), 'b-')   

plt.show()    