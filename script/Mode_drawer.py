"""

Script is to get points to be imported into sigmaplot 
No physics. 

"""

import numpy as np
from bmode import exact_mode

ndots = 100

atoms = {
            'Cesium': 2*np.pi*200/850,
            'Rubidium': 2*np.pi*200/780
        }



for Atom, a in atoms.items():
    xr = np.linspace(0.01, 5, ndots)*a
    
    args = {'k':1, 
            'n':1.45,
            'a': a
            }
    
    m = exact_mode(**args)
    
    f_cyl = open(Atom+'_cyl.txt', 'w')
    f_sph = open(Atom+'_sph.txt', 'w')
    
    f_cyl.write('x; E_r; E_phi; E_z;  \n') #Headers
    f_sph.write('x; E; dE; E_z;  \n')
    
    for x in xr:
        f_cyl.write(str(x/a)+';'+str(np.real(1j*m.Er(x))) + ';' + str(np.real(m.Ephi(x))) + ';'+str(np.real(m.Ez(x)))+'; \n')
        f_sph.write(str(x/a)+';'+str(np.real(-1j*m.E(x))) + ';' + str(np.real(1j*m.dE(x))) + ';'+str(np.real(m.Ez(x)))+'; \n')
        
    f_cyl.close()
    f_sph.close()
    
print('Written')
from matplotlib import pyplot as plt
args = {'k':1,
        'n':1.45,
        'a': 2*np.pi*200/850
       }
m = exact_mode(**args)
x = np.linspace(0,4, ndots)*2*np.pi*200/780
xin = np.linspace(0,0.95, ndots)*2*np.pi*200/780
xout = np.linspace(1.05,4, ndots)*2*np.pi*200/780

plt.plot(xin/a, abs(m.Er(xin)), 'r', label = '$|E_r|$', lw =1.5)
plt.plot(xout/a, abs(m.Er(xout)), 'r', lw =1.5)
plt.plot(x/a, abs(m.Ephi(x)), 'b',  label = '$|E_{\phi}|$', lw=1.5)
plt.plot(x/a, abs(m.Ez(x)), 'g', label = '$|E_z|$', lw = 1.5)
plt.legend(frameon=False)
plt.savefig('Modes.svg')
plt.show()
        
    
    
    