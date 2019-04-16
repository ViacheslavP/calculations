import numpy as np
from scipy.integrate import quad
from scipy.special import jn
import matplotlib.pyplot as plt

def sintegrand(q, rho1, k, dz):
    return q* jn(0,q*rho1)*hf(q)*np.sin(1*dz*np.sqrt(k*k - q*q))/np.sqrt(k*k - q*q)

def cintegrand(q, rho1, k, dz):
    return q* jn(0,q*rho1)*hf(q)*np.cos(1*dz*np.sqrt(k*k - q*q))/np.sqrt(k*k - q*q)

def hf(x):
    return x/(1+x*x)**(3/2)

nop = 1000
zs = np.linspace(0,100,nop)
ints = np.empty([nop])
imints = np.empty([nop])
absdif = np.empty([nop])
kmax = 2*np.pi
for i in range(nop):
    _sint = lambda x: sintegrand(x, 1.5, 2*np.pi, zs[i])
    _cint = lambda x: cintegrand(x, 1.5, 2 * np.pi, zs[i])
    ints[i] = (quad(_cint, 0, kmax)[0])
    imints[i] = (quad(_sint, 0, kmax)[0])
    absdif[i] = abs((ints[i]+1j*imints[i])/np.exp(1j*kmax*zs[i]))

plt.plot(zs,ints)
plt.plot(zs,imints)
plt.plot(zs, absdif)
plt.show()