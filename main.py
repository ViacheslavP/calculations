# -*- coding: utf-8 -*-
"""
Created 23.07.16

AS Sheremet's matlab code implementation 

n = 1.45?

UNSOLVED:
Selection rules


FUTURE:
Random ensemble(? idk how to), 
Use GPU for sparse matrix linsolve (inverse function could be rewritten on cuda
in nopython flags)


"""

try:
    import mkl
    mkl.set_num_threads(mkl.get_max_threads())
    mkl.service.set_num_threads(mkl.get_max_threads())
except ImportError:
    print('MKL does not installed. MKL dramatically improves calculation time')
    
    
from scipy.sparse import linalg as alg
from scipy.sparse import csr_matrix as csc
import numpy as np
from time import time


def inverse(ResInv,ddRight):
    return alg.spsolve(csc(ResInv),ddRight)

class ensemble(object):
    
        """
        Class of atom ensemble
        """
        def __init__(self):
        
            self.rr = np.array([[]])
            self.phib = np.array([],dtype=complex)
            self.xm = np.array([[]],dtype=complex)
            self.xp = np.array([[]],dtype=complex)
            self.x0 = np.array([[]],dtype=complex)
            self.index = np.array([[]])
            self.D = np.zeros([nat*3**nb,nat*3**nb],dtype=complex)
            #self.CrSec = np.zeros(nsp);
            self.Transmittance = np.zeros(nsp,dtype=complex);
            self.Reflection = np.zeros(nsp,dtype=complex);
            self._foo = [[[]]]
        def generate_ensemble(self, s='chain'):
    
            """
            Method generates enseble for current atomic ensemble properties and
            finds requested number of neighbours
            Yet it is a linear ensemble
            """  
            if s=='chain':
                x = 1.5*a*np.ones(nat)
                y = 0.*np.ones(nat)
                z = np.arange(nat)*Lz1
            elif s=='doublechain':
                x = 1.5*a*np.asarray([np.sign(i - nat/2) for i in range(nat)])
                y = 0.*np.ones(nat)
                goo = [i for i in range(nat//2)]
                z = np.asarray(goo+goo)
            else:
                raise NameError('Ensemble type not found')
        
            t1 = time()
            
            self.phib = 1/np.sqrt(2*np.pi*wb**2)*np.exp(((x)**2+y**2)/2/wb/wb)        
            
            self.rr = np.array(np.sqrt([[((x[i]-x[j])**2+(y[i]-y[j])**2+(z[i] \
            -z[j])**2)for i in range(nat)] for j in range(nat)]))
        
            self.xm = np.array([[((x[i]-x[j])-1j*(y[i]-y[j]))/(self.rr[i,j] + \
            np.identity(nat)[i,j]) for j in range(nat)]for i in range(nat)])
        
            self.xp = -np.array([[((x[i]-x[j])+1j*(y[i]-y[j]))/(self.rr[i,j]+ \
            np.identity(nat)[i,j]) for j in range(nat)]for i in range(nat)])
        
            self.x0 = np.array([[(z[i]-z[j])/(self.rr[i,j] + \
            np.identity(nat)[i,j]) for j in range(nat)] for i in range(nat)])
            
            self.index = np.argsort(self.rr)[:,1:nb+1]
            self.create_D_matrix()
            self.reflection_calc()
            
            print('Ensemble was generated for ',time()-t1,'s')
            
            
        def create_D_matrix(self):
            """
            Method creates st-matrix, then creates D matrix and reshapes D 
            matrix into 2x2 matrix with respect to selection rules
            

            
            We consider situation when the atom interacts just with some neighbours.
            It's mean that we should consider St-matrix only for this neighbours. For
            all other atoms we consider only self-interaction and elements of
            St-matrix would be the same for this elements.
            0 - atom is in state with m = -1,
            1 - atom is in state with m = 0,
            2 - atom is in state with m = 1. We assume that it t=0 all atoms in state
            with m = 1.
            """
            from itertools import product as combi
            state = np.asarray([i for i in combi(range(3),repeat=nb)]) 
            
            st = np.ones([nat,nat,3**nb])*2
            for n1 in range(nat):
                k=0
                for n2 in self.index[n1,:]:
                    for i in range(3**nb):
                        st[n1,n2,i] = state[i,k]
                    k+=1  
            """
            condition of selection for nb = 3
            st[n1,:,i] = [... 2, i0, 2(n1'th position) ,i1(n2'th position), i2   3...] 
            st[n2,:,j] = [... 2, 2, j0(n1'th position), 2(n2'th position),  j1, j2 ..]
            
            ik = state[i,k]
            
            the condition of transition from i to j is j1 = i2, j2 = 3 
            (we exclude n1't and n2'th positions)
            """

            def foo(n1,n2,i,j):

                for k1 in range(nat):
                    if k1 == n1 or k1 == n2: continue;
                    if (st[n1,k1,i]!=st[n2,k1,j]):
                        return False
                return True
                
                
            D1 = hbar*kv*((1 - 1j*self.rr - self.rr**2)/(((kv**3)*(self.rr+\
            np.identity(nat))**3)*np.exp(-1j*self.rr))*(np.ones(nat)-np.identity(nat)))   
            D2 = -hbar*kv*((3 - 3*1j*self.rr - self.rr**2)/(((kv**3)*(self.rr-\
            lambd*np.identity(nat))**3))*np.exp(1j*self.rr))
            D3 = np.zeros([nat,nat], dtype=complex)            
            for i in range(nat):
                for j in range(nat):
                    D3[i,j] = kv*(1/vg-1/c)*self.phib[i]*self.phib[j]*\
                    np.exp(1j*kv*self.x0[i,j]*self.rr[i,j])  

            Di = np.zeros([nat,nat,3,3], dtype = complex)
            Di[:,:,0,0] = d01m*d1m0*(D1 - self.xp*self.xm*D2);
            Di[:,:,0,1] = d01m*d00*(-self.xp*self.x0*D2-D3);
            Di[:,:,0,2] = d01m*d10*(-self.xp*self.xp*D2);
            Di[:,:,1,0] = d00*d1m0*(self.x0*self.xm*D2-D3);
            Di[:,:,1,1] = d00*d00*(D1 + self.x0*self.x0*D2); #not sure abou sign of D3
            Di[:,:,1,2] = d00*d10*(self.x0*self.xp*D2);
            Di[:,:,2,0] = d01*d1m0*(-self.xm*self.xm*D2);
            Di[:,:,2,1] = d01*d00*(-self.xm*self.x0*D2);
            Di[:,:,2,2] = d01*d10*(D1 - self.xm*self.xp*D2);
            
                    
            for n1 in range(nat):              #choose excited atom
                k = 0
                for n2 in self.index[n1,:]:    #choose neighbour
                    for i in range(3**nb):     #select excited atom environment
                        for j in range(3**nb): #select neighbour environment
                            if foo(n1,n2,i,j): #if transition is possible then make assigment
                                self.D[n1*3**nb+i,n2*3**nb+j] =  \
                                Di[n1,n2,state[i,k],state[j,k]]
                    k+=1            
            
        def reflection_calc(self):
            
            ddLeftF = np.zeros(nat*3**nb, dtype=complex);
            ddLeftB = np.zeros(nat*3**nb, dtype=complex);
            ddRight = np.zeros(nat*3**nb, dtype=complex);   
            for i in range(nat):
                ddRight[i*3**nb] = 1j*d10*np.exp(1j*kv*self.x0[0,i]*self.rr[0,i])*self.phib[i]
            
            Sigma = np.zeros([nat*3**nb,nat*3**nb]);

            
            for k in range(nsp):
                
                omega = deltaP[k]
                Sigma = np.identity(nat*3**nb)*(omega+1j*nat*g/2)
                
                ResInv = Sigma - self.D*kv*kv/hbar
                Resolventa = np.zeros(nat*nb**3)
                Resolventa =  inverse(ResInv,ddRight)
                
                TF = np.zeros(nat*3**nb, dtype=complex)
                TB = np.zeros(nat*3**nb, dtype=complex)
                
                for s in range(3):
                    
                    ddLeftF[s*3**nb] = -1j*d01*np.exp(-1j*kv*self.x0[0,s]*self.rr[0,s])*self.phib[s]
                    ddLeftB[s*3**nb] = -1j*d01*np.exp(-1j*kv*self.x0[0,s]*self.rr[0,s])*self.phib[s]
                    
                    TF[s*3**nb] =  2*np.pi*hbar*kv*c/Lz*Resolventa[s*3**nb]\
                    *ddLeftF[s*3**nb]
                    TB[s*3**nb] = 2*np.pi*hbar*kv*c/Lz*Resolventa[s*3**nb]\
                    *ddLeftF[s*3**nb]
                self.Transmittance[k] = 1+(-1j/hbar*np.sum(TF)*(Lz/vg))
                self.Reflection[k] = (-1j/hbar*np.sum(TF)*(Lz/vg))
                
        def visualize(self):
            import matplotlib.pyplot as plt
            plt.subplot(1,3,1)            
            plt.plot(deltaP,(abs(self.Reflection)**2 ), 'r-')
            plt.title('Reflection')
            plt.ylabel('$|r|^2$')
            
            plt.subplot(1,3,2) 
            plt.plot(deltaP, abs(self.Transmittance)**2, 'm-')
            plt.title('Transmittance')
            plt.ylabel('$|t|^2$')
            
            plt.subplot(1,3,3)
            plt.plot(deltaP, np.ones(nsp)-(abs(self.Reflection)**2\
            +abs(self.Transmittance)**2), 'g-')
            plt.title('Loss')
            plt.ylabel('$1-|r|^2-|t|^2$')
            
            plt.show()


"""
_____________________________________________________________________________
Declaring global variables
_____________________________________________________________________________

"""

#scale constants


hbar = 1 #dirac's constant
c = 1 #speed of light
gd = 1; #decay rate in vacuum
lambd = 1; #atomic wavelength (lambda=lambda/(2*pi))
lambd0 = lambd/2; #period of atomic chain
a = 200/780*lambd; #nanofiber radius in units of Rb wavelength


#dipole moments (?)

d01 = np.sqrt(hbar*gd/4*(lambd**3));  #|F = 0, n = 0> <- |F0 = 1> - populated level 
d10 = d01;
d00 = np.sqrt(hbar*gd/4*lambd**3);
d01m = np.sqrt(hbar*gd/4*lambd**3);
d1m0 = d01m;

#waveguide properties that could be computed in present scale

vg = 0.83375 #group velocity
g = 1.06586;  #decay rate corrected in presence of a nanofiber, units of gamma 
kv = 1.09629/lambd; #propogation constant (longitudial wavevector)
wb = 1.5

#atomic ensemble properties

n0 = 1*lambd**(-3); #density
nat = 35; #number of atoms
if nat%2 == 1: nat+=1
nb = 3;#number of neighbours
Lz1 = lambd0 #minimized size between neighbours 
Lz = 1.
#frequency detuning scale 

deltaP = np.arange(-10, 10, 1)*gd
nsp = len(deltaP);
#V = nat/n0;   %quantization volume


"""
______________________________________________________________________________
Executing program
______________________________________________________________________________
"""

chi = ensemble()
chi.generate_ensemble('doublechain')
chi.visualize()


        
