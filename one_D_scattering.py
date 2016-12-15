# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 14:00:47 2016

@author: viacheslav
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 18:47:05 2016

@author: viacheslav
"""

# -*- coding: utf-8 -*-
"""
Created 23.07.16

AS Sheremet's matlab code implementation 

n = 1.45 - the case that we only need to calculate
n = 1.1 - the other case

UNSOLVED:
|t|^2 ~ N^2 (as N->inf and as vacuum prop is on)


FUTURE:
V - atom (Self-Energy part needs to be modified)





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

#from cudasolve import cuSparseSolve as cu


        

def inverse(ResInv,ddRight):
    return alg.spsolve(csc(ResInv),(ddRight))
    #return cu(csc(ResInv),csc(ddRight)).toarray()

class ensemble(object):
    
        """
        Class of atom ensemble
        """
        def __init__(self,
                     nat=6,
                     nb=5,
                     s='chain',
                     dist=0.,
                     d = 1.5,
                     l0=2*np.pi,
                     deltaP=np.asarray([0.]),
                     typ='L',
                     dens=0.1
                     ):
                         
            self.rr = np.array([[]])
            self.phib = np.array([],dtype=complex)
            self.phibex = np.array([],dtype=complex)
            self.ep = np.array([],dtype=complex)
            self.em = np.array([],dtype=complex)
            self.ez = np.array([],dtype=complex)
            self.ezm = np.array([],dtype=complex)
            self.epm = np.array([],dtype=complex)
            self.xm = np.zeros([nat,nat],dtype=complex)
            self.xp = np.array([[]],dtype=complex)
            self.x0 = np.array([[]],dtype=complex)
            self.index = np.array([[]])
            self.D =  np.zeros([nat,nat],dtype=complex)
            self.Transmittance = np.zeros(len(deltaP),dtype=complex);
            self.Reflection = np.zeros(len(deltaP),dtype=complex);
            self.CrSec = np.zeros(len(deltaP),dtype=complex)
            self._foo = [[[]]]
            self.g = 1.06586/c;  #decay rate corrected in presence of a nanofiber, units of gamma 
            
            from bmode import exact_mode
            m = exact_mode(1/lambd, n0, a)
            self.vg = m.vg*c #group velocity
            self.kv = m._beta 
            self.E = m.E
            self.dE = m.dE
            self.Ez = m.Ez
            self.wb = 4.
            
            self._dist = dist
            self._s = s
            self.d = d
            self.nat = nat
            self.nb = nb
            self.deltaP=deltaP
            self.step=l0
            self.dens = dens
            self.typ = typ

            self.r = 1

            
        def generate_ensemble(self,dspl = True):
    
            """
            Method generates enseble for current atomic ensemble properties and
            finds requested number of neighbours
            Yet it is a linear ensemble
            """  
            step = self.step
            nat=self.nat
            s = self._s
            dist = self._dist
            nb = self.nb
            

            if s=='chain':
                x = self.d*a*np.ones(nat)
                y = 0.*np.ones(nat)
                z = np.arange(nat)*step
            elif s=='nocorrchain':
                x = self.d*a*np.ones(nat)
                y = 0.*np.ones(nat)
                L = nat*self.step
                z = np.random.rand(nat)*L
            elif s=='doublechain':
                x = self.d*a*np.asarray([np.sign(i - nat/2) for i in range(nat)])
                y = 0.*np.ones(nat)
                goo = [i for i in range(nat//2)]
                z = np.asarray(goo+goo)*step
                
            else:
                raise NameError('Ensemble type not found')
            
            if dist != .0 :
                pass
                x+= np.random.normal(0.0, dist*lambd, nat)
                y+= np.random.normal(0.0, dist*lambd, nat)
                z+= np.random.normal(0.0, dist*lambd, nat)
            
            

            t1 = time()
            
            if True:
                ga = lambda x,w: 1/np.sqrt(np.pi*w**2)*np.exp(-x**2/w**2/2)    
                self.phibex = -self.E(np.sqrt(x**2+y**2)) #for lambda-atom
                
                self.phib = ga(np.sqrt(x**2+y**2),self.wb)
                
                """
                Exact mode components
                e stands for exact mode
                m,p,z stands for component
                m stands for + \ phi (otherwise - \phi)
                
                """

                self.em = -self.E(np.sqrt(x**2+y**2))
                
                self.ep = self.dE(np.sqrt(x**2+y**2))*(x-1j*y)**2/(x**2+y**2)
                self.ez = self.Ez(np.sqrt(x**2+y**2))*(x-1j*y)/np.sqrt(x**2+y**2)
                
                self.epm = self.dE(np.sqrt(x**2+y**2))*(x+1j*y)**2/(x**2+y**2)
                self.ezm = self.Ez(np.sqrt(x**2+y**2))*(x+1j*y)/np.sqrt(x**2+y**2)
            
            self.r = np.sqrt(x**2+y**2)
            self.rr = np.asarray(np.sqrt([[((x[i]-x[j])**2+(y[i]-y[j])**2+(z[i] \
            -z[j])**2)for i in range(nat)] for j in range(nat)]),dtype=complex)
            self.xm = np.asarray([[((x[i]-x[j])-1j*(y[i]-y[j]))/(self.rr[i,j] + \
            np.identity(nat)[i,j]) for j in range(nat)]for i in range(nat)],dtype=complex)*(1/np.sqrt(2))
        
            self.xp = -np.asarray([[((x[i]-x[j])+1j*(y[i]-y[j]))/(self.rr[i,j]+ \
            np.identity(nat)[i,j]) for j in range(nat)]for i in range(nat)],dtype=complex)*(1/np.sqrt(2))
        
            self.x0 = np.asarray([[(z[i]-z[j])/(self.rr[i,j] + \
            np.identity(nat)[i,j]) for j in range(nat)] for i in range(nat)],dtype=complex)
            
            self.index = np.argsort(self.rr)[:,1:nb+1]
            
            self.create_D_matrix()
            self.reflection_calc()

            print('Ensemble was generated for ',time()-t1,'s')
            
        

        def create_D_matrix(self):
            """
            Method creates st-matrix, then creates D matrix and reshapes D 
            matrix into 1+1 matrix with respect to selection rules
            

            
            We consider situation when the atom interacts just with some neighbours
            (in Raman chanel!).
            It's mean that we should consider St-matrix only for this neighbours. For
            all other atoms we consider only self-interaction and elements of
            St-matrix would be the same for this elements.
            0 - atom is in state with m = -1,
            1 - atom is in state with m = 0,
            2 - atom is in state with m = 1. We assume that it t=0 all atoms in state
            with m = 1.
            """
            nat = self.nat
            nb = self.nb
            kv = 1./lambd
            


            """
            condition of selection for nb = 3
            st[n1,:,i] = [... 2, i0, 2(n1'th position) ,i1(n2'th position), i2   3...] 
            st[n2,:,j] = [... 2, 2, j0(n1'th position), 2(n2'th position),  j1, j2 ..]
            
            ik = state[i,k]
            
            the condition of transition from i to j is j1 = i2, j2 = 3 
            (we exclude n1'th and n2'th positions)
            n2 could be everywhere in ensemble
            """

            def foo(ni1,nj2,i,j):

                for ki1 in np.append(self.index[ni1,:],self.index[nj2,:]):
                    if ki1 == ni1 or ki1 == nj2: continue;
                    if (st[ni1,ki1,i] != st[nj2,ki1,j]):
                        return False
                return True

            L = 3 * np.pi / 1
            def foo2(a):
                if a > L:
                    return 0
                else:
                    return 1
            neigh = np.zeros([nat,nat])
            for i in range(nat):
                for j in range(nat):
                    neigh[i,j] = foo2(self.rr[i,j])

            kv = 1
            
            D1 = 0*kv*((1 - 1j*self.rr*kv - (self.rr*kv)**2)/ \
                ((self.rr*kv + np.identity(nat))**3) \
                *np.exp(1j*self.rr*kv)) *(np.ones(nat)-np.identity(nat))   
            D2 = 0*-1*hbar*kv*((3 - 3*1j*self.rr*kv - (self.rr*kv)**2)/((((kv*self.rr+\
            np.identity(nat))**3))*np.exp(1j*kv*self.rr)))
            
            Dz = np.zeros([nat,nat], dtype=complex)
            if True:
                for i in range(nat):
                    for j in range(nat):
    
                        Dz[i,j] = +1*2j*np.pi*kv*np.exp(1j*self.kv*(abs(self.x0[i,j]))* \
                        self.rr[i,j])*(1/self.vg)

            #Interaction part
            Di = np.zeros([nat,nat,3,3], dtype = complex)

            
            for i in range(nat):
                for j in range(nat):
                        
                    Di[i,j,0,0] = d01m*d1m0*(D1[i,j] -\
                                            self.xp[i,j]*self.xm[i,j]*D2[i,j]-\
                                            (np.conjugate(self.ep[i])*self.ep[j]+\
                                            self.em[i]*np.conjugate(self.em[j]))*Dz[i,j])
                    
                    Di[i,j,0,1] = d01m*d00*(-self.xp[i,j]*self.x0[i,j]*D2[i,j] - \
                                            (np.conjugate(self.ep[i])*(self.ez[j])-\
                                            self.em[i]*np.conjugate(self.ezm[j]))*Dz[i,j]);
                                            
                    Di[i,j,0,2] = d01m*d10*(-self.xp[i,j]*self.xp[i,j]*D2[i,j] -\
                                            (np.conjugate(self.ep[i])*(self.em[j])+\
                                             self.em[i]*np.conjugate(self.epm[j]))*Dz[i,j]);
                                            
                    Di[i,j,1,0] = d00*d1m0*(self.x0[i,j]*self.xm[i,j]*D2[i,j] -\
                                            (np.conjugate(self.ez[i])*(self.ep[j])-\
                                            self.ez[i]*np.conjugate(self.em[j]))*Dz[i,j]);
                                            
                    Di[i,j,1,1] = d00*d00*(D1[i,j] + \
                                           self.x0[i,j]*self.x0[i,j]*D2[i,j]-\
                                           (np.conjugate(self.ez[i])*(self.ez[j])+\
                                           self.ez[i]*np.conjugate(self.ez[j]))*Dz[i,j]);
                                           
                    Di[i,j,1,2] = d00*d10*(self.x0[i,j]*self.xp[i,j]*D2[i,j] - \
                                           (np.conjugate(self.ez[i])*(self.em[j])-\
                                           self.ez[i]*np.conjugate(self.epm[j]))*Dz[i,j]);
                                            
                    Di[i,j,2,0] = d01*d1m0*(-self.xm[i,j]*self.xm[i,j]*D2[i,j] -\
                                            (np.conjugate(self.em[i])*(self.ep[j])+\
                                            self.ep[i]*np.conjugate(self.em[j]))*Dz[i,j]);
                                            
                    Di[i,j,2,1] = d01*d00*(-self.xm[i,j]*self.x0[i,j]*D2[i,j]-\
                                            (np.conjugate(self.em[i])*(self.ez[j])-\
                                            self.ep[i]*np.conjugate(self.ez[j]))*Dz[i,j]);
                                            
                    Di[i,j,2,2] = d01*d10*(D1[i,j]- \
                                            self.xm[i,j]*self.xp[i,j]*D2[i,j]- \
                                            (np.conjugate(self.em[i])*(self.em[j])+\
                                            self.ep[i]*np.conjugate(self.epm[j]))*Dz[i,j])
            
            print(np.linalg.eigvals(Di[0,0,:,:]))
            #Basis part              
            if self.typ == 'L': 
                from itertools import product as combi
                state = np.asarray([i for i in combi(range(3),repeat=nb)]) 
                st = np.ones([nat,nat,3**nb], dtype = int)*2
                for n1 in range(nat):
                    k=0
                    for n2 in np.sort(self.index[n1,:]):
                        for i in range(3**nb):
                            st[n1,n2,i] = state[i,k]
                        k+=1  
                #self.D = lm((nat*3**nb,nat*3**nb),dtype=complex)                          
                self.D = np.zeros([3**nb*nat,3**nb*nat],dtype=complex)

                for n1 in range(nat):
                    for n2 in range(nat):#choose initial excited atom
                        for i in range(3**nb):  #select initial excited atom environment
                        #choose final excited or np.sort(self.index[n1,:]): if massive photon
                              
                            for j in range(3**nb):      #select final excited environment
                                if foo(n1,n2,i,j):  #if transition is possible then make assigment
                                    self.D[n1*3**nb+i,n2*3**nb+j] =  \
                                    Di[n1,n2,st[n1,n2,i],st[n2,n1,j]]

                                    
            elif self.typ == 'V':
                self.D = np.zeros([3*nat,3*nat],dtype=complex)
                for n1 in range(nat): #initial excited
                    
                    for n2 in range(nat):#final excited
                        for i in range(3):
                        
                        
                            for j in range(3):
                                self.D[3*n1+i,3*n2+j] = 3*Di[n1,n2,i,j]
            else:
                raise NameError('No such type')

            from matplotlib.pylab import spy,show
            spy(self.D) # Visualise matrix
            #spy(neigh)
            show()

            
        def reflection_calc(self):
            import sys
            nat=self.nat
            nb = self.nb
            k=1
            kv=1
            c=1
            

            gd = 1+8*np.pi*k*(1/self.vg-0/c)*self.phib*self.phib*d00*d00
            
            if self.typ == 'L':
                
                ddLeftF = np.zeros(nat*3**nb, dtype=complex);
                ddLeftB = np.zeros(nat*3**nb, dtype=complex);
                ddRight = np.zeros(nat*3**nb, dtype=complex);  
                One = (np.identity(nat*3**nb)) # Unit in Lambda-atom with nb neighbours
                Sigmad = (np.identity(nat*3**nb,dtype=complex))
                
                for i in range(nat):
                    ddRight[(i+1)*3**nb-1] = 1j*d10*np.exp(-1j*self.kv*self.x0[0,i]*self.rr[0,i])*self.phibex[i]
                    ddLeftF[(i+1)*3**nb-1] = -1j*d01*np.exp(+1j*self.kv*self.x0[0,i]*self.rr[0,i])*self.phibex[i]
                    ddLeftB[(i+1)*3**nb-1] = -1j*d01*np.exp(-1j*self.kv*self.x0[0,i]*self.rr[0,i])*self.phibex[i]
                    for j in range(3**nb):
                        Sigmad[(i)*3**nb+j,(i)*3**nb+j] = gd[i]*0.5j #distance dependance of dacay rate
                    
            elif self.typ == 'V':
                
                
                
                ddLeftF = np.zeros(3*nat, dtype=complex);
                ddLeftB = np.zeros(3*nat, dtype=complex);
                ddRight = np.zeros(3*nat, dtype=complex); 
                One = (np.identity(3*nat)) #Unit in V-atom
                Sigmad = (np.identity(nat*3,dtype=complex))
                for i in range(nat):

                    #--In-- chanel
                    ddRight[3*i] = np.sqrt(3)*1j*d10*np.exp(-1j*self.kv*self.x0[0,i]*self.rr[0,i])  \
                    *self.em[i]
                    ddRight[3*i+1] = -np.sqrt(3)*1j*d10*np.exp(-1j*self.kv*self.x0[0,i]*self.rr[0,i])  \
                    *self.ez[i] 
                    ddRight[3*i+2] = np.sqrt(3)*1j*d10*np.exp(-1j*self.kv*self.x0[0,i]*self.rr[0,i])  \
                    *self.ep[i]
                    
                    #-- Out Forward --
                    ddLeftF[3*i] = np.sqrt(3)*-1j*d01*np.exp(+1j*self.kv*self.x0[0,i]*self.rr[0,i]) \
                    *np.conjugate(self.em[i])
                    ddLeftF[3*i+1] = -np.sqrt(3)*-1j*d01*np.exp(+1j*self.kv*self.x0[0,i]*self.rr[0,i]) \
                    *np.conjugate(self.ez[i])
                    ddLeftF[3*i+2] = np.sqrt(3)*-1j*d01*np.exp(+1j*self.kv*self.x0[0,i]*self.rr[0,i]) \
                    *np.conjugate(self.ep[i])
                    
                    # -- Out Backward --
                    ddLeftB[3*i] = np.sqrt(3)*-1j*d01*np.exp(-1j*self.kv*self.x0[0,i]*self.rr[0,i]) \
                    *np.conjugate(self.em[i])
                    ddLeftB[3*i+1] = -np.sqrt(3)*-1j*d01*np.exp(-1j*self.kv*self.x0[0,i]*self.rr[0,i]) \
                    *np.conjugate(self.ez[i])
                    ddLeftB[3*i+2] = np.sqrt(3)*-1j*d01*np.exp(-1j*self.kv*self.x0[0,i]*self.rr[0,i]) \
                    *np.conjugate(self.ep[i])
                    
           
                    for j in range(3):
                        if j == 1: 
                            a = 1
                        else:
                            a = 0
                            
                        Sigmad[(i)*3+j,(i)*3+j] =0*(-0.0355+0.0*3*a*0.5*(gd[i]-1))*0.5j 
               
               #distance dependance of dacay rate
               # there is no in/out gaussian chanel  
               # for i in range(nat):
               #     ddRight[3*i] = np.sqrt(3)*1j*d10*np.exp(-1j*self.kv*self.x0[0,i]*self.rr[0,i])*self.phibex[i]
               #     ddLeftF[3*i] = np.sqrt(3)*-1j*d01*np.exp(+1j*self.kv*self.x0[0,i]*self.rr[0,i])*self.phibex[i]
               #     ddLeftB[3*i] = np.sqrt(3)*-1j*d01*np.exp(-1j*self.kv*self.x0[0,i]*self.rr[0,i])*self.phibex[i]
               #     for j in range(3):
               #         Sigmad[(i)*3+j,(i)*3+j] =(1+0.5*(gd[i]-1))*0.5j #distance dependance of dacay rate

            for k in range(len(self.deltaP)):
                
                omega = self.deltaP[k]
                 
                #V = lm(np.zeros([nat*3**nb,nat*3**nb], dtype = complex))
                Sigma = Sigmad+omega*One
                V =  1*(- self.D/hbar/lambd/lambd )
                ResInv = Sigma + V   
                Resolventa =  np.dot(np.linalg.inv(ResInv),ddRight)
                TF_ = np.dot(Resolventa,ddLeftF)*2*np.pi*hbar*kv
                TB_ = np.dot(Resolventa,ddLeftB)*2*np.pi*hbar*kv
                
                
                self.Transmittance[k] = 1.+(-1j/hbar*(TF_)/(self.vg))
                self.Reflection[k] = (-1j/hbar*(TB_/(self.vg)))
                
                
                    
                
                ist = 100*k / len(self.deltaP)
                sys.stdout.write("\r%d%%" % ist)
                sys.stdout.flush()
            print('\n')
            
            
            
        def visualize(self,addit='',save=True):
            import matplotlib.pyplot as plt
            
            plt.plot(self.deltaP,(abs(self.Reflection)**2 ), 'r-',label='R',lw=1.5)
            plt.plot(self.deltaP, abs(self.Transmittance)**2, 'm-', label='T',lw=1.5)
            plt.legend()
            plt.show()
            
            plt.title('Loss')
            plt.plot(self.deltaP, 1-(abs(self.Reflection)**2\
            +abs(self.Transmittance)**2), 'g-',lw=1.5)
            plt.show()
         
            
 #scale constants


hbar = 1 #dirac's constant = 1 => enrergy = omega
c = 1 #speed of light c = 1 => time in cm => cm in wavelenght
gd = 1.; #vacuum decay rate The only 
lambd = 1; # atomic wavelength (lambdabar)
k = 1/lambd
lambd0 = 1.*np.pi  /1.0951842440279331; # period of atomic chain
a = 2*np.pi*200/850*lambd; # nanofiber radius in units of Rb wavelength
n0 = 1.45 #(silica) 



#dipole moments (Wigner-Eckart th. !!), atom properies

d01 = np.sqrt(hbar*gd/4*(lambd**3));  #|F = 0, n = 0> <- |F0 = 1> - populated level 
d10 = d01;
d00 = np.sqrt(hbar*gd/4*lambd**3);
d01m = np.sqrt(hbar*gd/4*lambd**3);
d1m0 = d01m;


#atomic ensemble properties

freq = np.linspace(-1, 1, 100)*gd
step = lambd0


"""
______________________________________________________________________________
Executing program
______________________________________________________________________________

"""


if __name__ == '__main__':
    args = {
            'nat':1, #number of atoms
            'nb':5, #number of neighbours in raman chanel (for L-atom only)
            's':'chain', #Stands for atom positioning : chain, nocorrchain and doublechain
            'dist':0,  # sigma for displacement (choose 'chain' for gauss displacement.)
            'd' : 1.5, # distance from fiber
            'l0':lambd0, # mean distance between atoms (lambdabar units)
            'deltaP':freq,  # array of freq.
            'typ':'V',  # L or V for Lambda and V atom resp.
            }
            
    chi = ensemble(**args)
    chi.generate_ensemble()
    chi.visualize()
