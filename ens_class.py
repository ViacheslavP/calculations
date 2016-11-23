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
|t|^2 ~ N^2 (as N->inf)


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
from scipy.sparse import lil_matrix as lm
from scipy.sparse import lil

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
                     env='fiber',
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
            self.xm = np.zeros([nat,nat],dtype=complex)
            self.xp = np.array([[]],dtype=complex)
            self.x0 = np.array([[]],dtype=complex)
            self.index = np.array([[]])
            self.D =  np.zeros([nat,nat],dtype=complex)
            self.Transmittance = np.zeros(len(deltaP),dtype=complex);
            self.Reflection = np.zeros(len(deltaP),dtype=complex);
            self.CrSec = np.zeros(len(deltaP),dtype=complex)
            self._foo = [[[]]]
            if env == 'vacuum':
                self.vg = 1.
                self.kv=1/lambd
                self.g=1.
                self.wb = 100
            elif env == 'fiber':
                _s = "mode_data_%g_%g_%g" % (1/lambd,n0,a)
                import os.path
                if os.path.isfile(_s):
                    with open(_s, 'r') as f:
                        _fop = np.fromfile(f)
                        self.kv = _fop[0]
                        self.wb = _fop[1]
                        self.vg = _fop[2]
                        self.g = 1.06586*gd/c
                else:
                    from fundamental_mode import exact_mode
                    m = exact_mode(1/lambd, n0, a)
                
                    self.vg = m.vg*c #group velocity
                    self.g = 1.06586/c;  #decay rate corrected in presence of a nanofiber, units of gamma 
                    self.kv = m._beta #1.09629/lambd; #propogation constant (longitudial wavevector)
                    self.wb = m.wb
            elif env == 'no correction':
                _s = "mode_data_%g_%g_%g" % (1/lambd,n0,a)
                import os.path
                if os.path.isfile(_s):
                    with open(_s, 'r') as f:
                        _fop = np.fromfile(f)
                        self.kv = _fop[0]
                        self.wb = _fop[1]
                        self.vg = _fop[2]
                        self.g = 1.0*gd/c
                else:
                    from fundamental_mode import exact_mode
                    m = exact_mode(1/lambd, n0, a)
                
                    self.vg = m.vg*c #group velocity
                    self.g = 1.06586/c;  #decay rate corrected in presence of a nanofiber, units of gamma 
                    self.kv = m._beta #1.09629/lambd; #propogation constant (longitudial wavevector)
                    self.wb = m.wb
            else:
                raise NameError('Enviroment not set')
            self._dist = dist
            self._s = s
            self._env = env
            self.d = d
            self.nat = nat
            self.nb = nb
            self.deltaP=deltaP
            self.step=l0
            self.dens = dens
            self.dens_r = 0.1
            self.typ = typ
            self.sigma = 1.
            
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
            if self._env == 'vacuum':
            
                  
                R = (nat / (self.dens * 4*np.pi/3))**(1/3)
                rfoo = lambda x,y,z: np.sqrt(x*x+y*y+z*z)
                while True:                
                    rad = np.random.rand(nat)*R
                    phi = 2*np.pi*np.random.rand(nat)
                    theta = np.pi*np.random.rand(nat)
                    x = np.zeros(nat); y = np.zeros(nat); z = np.zeros(nat)                    
                    for i in range(nat):
                        x[i] = rad[i]*np.cos(phi[i])*np.sin(theta[i])
                        y[i] = rad[i]*np.sin(phi[i])*np.sin(theta[i])
                        z[i] = rad[i]*np.cos(theta[i])
                    tp = True    
                    for i in range(nat):
                        for j in range(i):
                            if i==j:
                                continue
                                
                            if rfoo(x[i]-x[j],y[i]-y[j],z[i]-z[j]) < 0.05:                       
                                tp = False
                                
                    if tp:
                        break
                    
                #x = np.array([0,0.5/np.sqrt(2)])
                #y = np.array([0,0])
                #z = np.array([0,0.5/np.sqrt(2)])
                
                from mpl_toolkits.mplot3d import Axes3D
                import matplotlib.pyplot as plt
 

                
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                 
                ax.scatter(x, y, z, s=np.pi*8, c='blue', alpha=0.75)
                 
                ax.set_xlabel('$x , \frac{\lambda}{2 \pi} $', fontsize=16)
                ax.set_ylabel('$y , \frac{\lambda}{2 \pi} $', fontsize=16)
                ax.set_zlabel('$z , \frac{\lambda}{2 \pi} $', fontsize=16)
                 
                plt.savefig('spheres.png')
                    
                plt.show()
                    
               

                    
            elif s=='chain':
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
            
            if dist != .0 :#and s!='frog':
                pass
                x+= np.random.normal(0.0, dist*lambd, nat)
                y+= np.random.normal(0.0, dist*lambd, nat)
                z+= np.random.normal(0.0, dist*lambd, nat)
            
            

            t1 = time()
            
            ga = lambda x,w: 1/np.sqrt(np.pi*w**2)*np.exp(-x**2/w**2/2)    
            self.phib = ga(x,self.wb)
            
            
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

            if self._env == 'fiber' or self._env == 'no correction':            
                self.reflection_calc()
            elif self._env == 'vacuum':
                self.crsec_calc()
            
            print('Ensemble was generated for ',time()-t1,'s')
            
        

        def create_D_matrix(self):
            """
            Method creates st-matrix, then creates D matrix and reshapes D 
            matrix into 1+1 matrix with respect to selection rules
            

            
            We consider situation when the atom interacts just with some neighbours.
            It's mean that we should consider St-matrix only for this neighbours. For
            all other atoms we consider only self-interaction and elements of
            St-matrix would be the same for this elements.
            0 - atom is in state with m = -1,
            1 - atom is in state with m = 0,
            2 - atom is in state with m = 1. We assume that it t=0 all atoms in state
            with m = 1.
            """
            nat=self.nat
            nb = self.nb
            kv = 1./lambd
            


            """
            condition of selection for nb = 3
            st[n1,:,i] = [... 2, i0, 2(n1'th position) ,i1(n2'th position), i2   3...] 
            st[n2,:,j] = [... 2, 2, j0(n1'th position), 2(n2'th position),  j1, j2 ..]
            
            ik = state[i,k]
            
            the condition of transition from i to j is j1 = i2, j2 = 3 
            (we exclude n1't and n2'th positions)
            n2 could be everywhere in ensemble
            """

            def foo(ni1,nj2,i,j):

                for ki1 in np.append(self.index[ni1,:],self.index[nj2,:]):
                    if ki1 == ni1 or ki1 == nj2: continue;
                    if (st[ni1,ki1,i] != st[nj2,ki1,j]):
                        return False
                return True
                
                
            D1 = ((1 - 1j*self.rr - self.rr**2)/ \
                ((self.rr + np.identity(nat))**3) \
                *np.exp(1j*self.rr)) *(np.ones(nat)-np.identity(nat))   / kv/kv
            D2 = -hbar*kv*((3 - 3*1j*self.rr - self.rr**2)/(((kv**3)*(self.rr+\
            lambd*np.identity(nat))**3))*np.exp(1j*self.rr))
            
            D3 = np.zeros([nat,nat], dtype=complex)            
            for i in range(nat):
                for j in range(nat):
                    D3[i,j] =-2*np.pi*1j* kv*(1/self.vg-1/c)*self.phib[i]*self.phib[j]*\
                    np.exp(1j*kv*self.x0[i,j]*self.rr[i,j]) #waveguide correction
                    if i==j: D3[i,j] = 0;
            if self._env == 'vacuum' or self._env == 'no correction':
                    D3 = np.zeros([nat,nat], dtype=complex) 
            #Self-Energy part
            #D1 = np.zeros([nat,nat], dtype=complex) 
            #D2 = np.zeros([nat,nat], dtype=complex) 
            Di = np.zeros([nat,nat,3,3], dtype = complex)


            for i in range(nat):
                for j in range(nat):
                        
                    Di[i,j,0,0] = d01m*d1m0*(D1[i,j]- self.xp[i,j]*self.xm[i,j]*D2[i,j]+D3[i,j]);
                    Di[i,j,0,1] = d01m*d00*(-self.xp[i,j]*self.x0[i,j]*D2[i,j]);
                    Di[i,j,0,2] = d01m*d10*(-self.xp[i,j]*self.xp[i,j]*D2[i,j]);
                    Di[i,j,1,0] = d00*d1m0*(self.x0[i,j]*self.xm[i,j]*D2[i,j]);
                    Di[i,j,1,1] = d00*d00*(D1[i,j] + self.x0[i,j]*self.x0[i,j]*D2[i,j]); 
                    Di[i,j,1,2] = d00*d10*(self.x0[i,j]*self.xp[i,j]*D2[i,j]);
                    Di[i,j,2,0] = d01*d1m0*(-self.xm[i,j]*self.xm[i,j]*D2[i,j]);
                    Di[i,j,2,1] = d01*d00*(-self.xm[i,j]*self.x0[i,j]*D2[i,j]);
                    Di[i,j,2,2] = d01*d10*(D1[i,j]- self.xm[i,j]*self.xp[i,j]*D2[i,j]+D3[i,j]);
             
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
            show()
            
        def reflection_calc(self):
            import sys
            nat=self.nat
            nb = self.nb
            k=1
            
            if self._env == 'fiber':
                gd = 1+8*np.pi*k*(1/self.vg-1/c)*self.phib*self.phib*d00*d00
            
            if self.typ == 'L':
                
                ddLeftF = np.zeros(nat*3**nb, dtype=complex);
                ddLeftB = np.zeros(nat*3**nb, dtype=complex);
                ddRight = np.zeros(nat*3**nb, dtype=complex);  
                One = (np.identity(nat*3**nb)) # Unit in Lambda-atom with nb neighbours
                Sigmad = (np.identity(nat*3**nb,dtype=complex))
                
                for i in range(nat):
                    ddRight[(i+1)*3**nb-1] = 1j*d10*np.exp(-1j*self.kv*self.x0[0,i]*self.rr[0,i])*self.phib[i]
                    ddLeftF[(i+1)*3**nb-1] = -1j*d01*np.exp(+1j*self.kv*self.x0[0,i]*self.rr[0,i])*self.phib[i]
                    ddLeftB[(i+1)*3**nb-1] = -1j*d01*np.exp(-1j*self.kv*self.x0[0,i]*self.rr[0,i])*self.phib[i]
                    for j in range(3**nb):
                        Sigmad[(i)*3**nb+j,(i)*3**nb+j] = gd[i]*0.5j #distance dependance of dacay rate
                    
            elif self.typ == 'V':
                
                ddLeftF = np.zeros(3*nat, dtype=complex);
                ddLeftB = np.zeros(3*nat, dtype=complex);
                ddRight = np.zeros(3*nat, dtype=complex); 
                One = lm(np.identity(3*nat)) #Unit in V-atom
                Sigmad = (np.identity(nat*3,dtype=complex))
                for i in range(nat):
                    ddRight[3*i] = np.sqrt(3)*1j*d10*np.exp(-1j*self.kv*self.x0[0,i]*self.rr[0,i])*self.phib[i]
                    ddLeftF[3*i] = np.sqrt(3)*-1j*d01*np.exp(+1j*self.kv*self.x0[0,i]*self.rr[0,i])*self.phib[i]
                    ddLeftB[3*i] = np.sqrt(3)*-1j*d01*np.exp(-1j*self.kv*self.x0[0,i]*self.rr[0,i])*self.phib[i]
                    for j in range(3):
                        Sigmad[(i)*3+j,(i)*3+j] =0.5* gd[i]*0.5j #distance dependance of dacay rate

            for k in range(len(self.deltaP)):
                
                omega = self.deltaP[k]
                V =  (- self.D/hbar/lambd/lambd )     
                #V = lm(np.zeros([nat*3**nb,nat*3**nb], dtype = complex))
                Sigma = Sigmad+omega*One
                ResInv = Sigma + V           
                Resolventa =  inverse(ResInv,ddRight )

                TF_ = np.dot(Resolventa,ddLeftF)*2*np.pi*hbar*self.kv
                TB_ = np.dot(Resolventa,ddLeftB)*2*np.pi*hbar*self.kv
                
                
                self.Transmittance[k] = 1.+(-1j/hbar*(TF_)/(self.vg))
                self.Reflection[k] = (1j/hbar*(TB_)/(self.vg))
                
                
                ist = 100*k / len(self.deltaP)
                sys.stdout.write("\r%d%%" % ist)
                sys.stdout.flush()
            print('\n')
            
        def crsec_calc(self):
            import sys
            nat=self.nat
            nb = self.nb
            k=1
            

            if self.typ == 'L':
                
                ddLeftF = np.zeros(nat*3**nb, dtype=complex);
                ddLeftB = np.zeros(nat*3**nb, dtype=complex);
                ddRight = np.zeros(nat*3**nb, dtype=complex);  
                

                for i in range(nat):
                    ddRight[(i+1)*3**nb-1] = 1j*d10*np.exp(-1j*self.kv*self.x0[0,i]*self.rr[0,i])
                    ddLeftF[(i+1)*3**nb-1] = -1j*d01*np.exp(+1j*self.kv*self.x0[0,i]*self.rr[0,i])
                    ddLeftB[(i+1)*3**nb-1] = -1j*d01*np.exp(-1j*self.kv*self.x0[0,i]*self.rr[0,i])
                    One = lm(np.identity(nat*3**nb, dtype=complex))
                
                
            elif self.typ == 'V':
                
                ddLeftF = np.zeros(3*nat, dtype=complex)
                ddLeftB = np.zeros(3*nat, dtype=complex)
                ddRight = np.zeros(3*nat, dtype=complex)  
                
                for i in range(nat):
                    ddRight[3*i] = np.sqrt(3)*1j*d10*np.exp(-1j*self.kv*self.x0[0,i]*self.rr[0,i])
                    ddLeftF[3*i] = np.sqrt(3)*-1j*d01*np.exp(+1j*self.kv*self.x0[0,i]*self.rr[0,i])
                    ddLeftB[3*i] = np.sqrt(3)*-1j*d01*np.exp(-1j*self.kv*self.x0[0,i]*self.rr[0,i])
                    One = lm(np.identity(3*nat,dtype=complex))
                    
                    
            for k in range(len(self.deltaP)):
                
                omega = self.deltaP[k]
                
                V =   -self.D*self.kv*self.kv/hbar     

                Sigma = One*(omega+1j*self.g/2)
                ResInv = Sigma + V           
                Resolventa =  inverse(ResInv,ddRight)
                
                
                TF_ = np.dot(Resolventa,ddLeftF)*2*np.pi*hbar*self.kv

                
                self.CrSec[k] = -2 * TF_.imag / hbar / c 
                ist = 100*k / len(self.deltaP)
                sys.stdout.write("\r%d%%" % ist)
                sys.stdout.flush()
                
            print('\n')    
            sys.stdout.flush()
                
                
                
        def vis_crsec(self):
            
            import matplotlib.pyplot as plt
            plt.plot(self.deltaP, np.real(self.CrSec), 'b', lw =2.)
            plt.xlabel('Detuning, '+'$\gamma$', fontsize=16)
            plt.ylabel('$ \sigma , \lambdabar ^2 $', fontsize=16)
            plt.savefig('CrSc_V_diag_0.5.png')
            plt.show()  
            
        def visualize(self,addit='',save=True):
            import matplotlib.pyplot as plt
            nat=self.nat
            plt.subplots(figsize=(14,20))
            plt.subplots_adjust(hspace=0.5)
            if self._env == 'fiber' :
                s = '$N_{a} = %i,\, Beam\, waist = %g \lambdabar,\, \sigma = %g \lambdabar ,\, Fiber\, ON$' % (nat,self.wb,self._dist)
            elif self._env == 'vacuum':
                s = '$N_{a} = %i,\, Beam\, waist = %g \lambdabar,\, \sigma = %g \lambdabar, \, Fiber\, OFF$' % (nat,self.wb,self._dist)
            else:
                raise NameError('Got problem with fiber')
            plt.suptitle(s,fontsize=20)
            
            plt.subplot(3,1,1)            
            plt.plot(self.deltaP,(abs(self.Reflection)**2 ), 'r-')
            plt.title('$Reflection$',fontsize=16)
            plt.ylabel('$|r|^2$',fontsize=16)
            plt.xlabel('$\delta \omega, \gamma$',fontsize=16)
            
            plt.subplot(3,1,2) 
            plt.plot(self.deltaP, abs(self.Transmittance)**2, 'm-')#abs(self.Transmittance)**2, 'm-')
            plt.title('$Transmission$',fontsize=16)
            plt.ylabel('$|t|^2$',fontsize=16)
            #plt.ylim(0,1)            
            plt.xlabel('$\delta \omega, \gamma$',fontsize=16)
            
            plt.subplot(3,1,3)
            plt.plot(self.deltaP, np.ones(len(self.deltaP))-(abs(self.Reflection)**2\
            +abs(self.Transmittance)**2), 'g-')
            plt.title('$Loss$',fontsize=16)
            plt.ylabel('$1-|r|^2-|t|^2$',fontsize=16)
            plt.xlabel('$\delta \omega, \gamma$',fontsize=16)
            if save:
                plt.savefig(('TRL_%i_%g_%g_%s' % (nat,self.wb,self._dist,self._env))+addit+'.png')
            else:
                plt.show()
            plt.show()
            
            plt.plot(self.deltaP,(abs(self.Reflection)**2 ), 'r-')
            plt.title('$Reflection$',fontsize=16)
            plt.ylabel('$|r|^2$',fontsize=16)
            plt.xlabel('$\delta \omega, \gamma$',fontsize=16)
            plt.savefig('Ref1.1.png')
            
            #k = [1-self.Transmittance[i].real/(abs(self.Reflection)[i]**2\
            #+abs(self.Transmittance[i])**2) for i in range(nsp)]
            #plt.plot(deltaP,k)
            #plt.show()
            
            #scale constants


hbar = 1 #dirac's constant = 1 => enrergy = omega
c = 1 #speed of light c = 1 => time in cm => cm in wavelenght
gd = 1.; #vacuum decay rate The only 
lambd = 1; # atomic wavelength (lambdabar)
k = 1/lambd
lambd0 = 1.*np.pi / 1.09; # period of atomic chain
a = 2*np.pi*200/780*lambd; # nanofiber radius in units of Rb wavelength
n0 = 1.45 #(silica) 



#dipole moments (Wigner-Eckart th. !!), atom properies

d01 = np.sqrt(hbar*gd/4*(lambd**3));  #|F = 0, n = 0> <- |F0 = 1> - populated level 
d10 = d01;
d00 = np.sqrt(hbar*gd/4*lambd**3);
d01m = np.sqrt(hbar*gd/4*lambd**3);
d1m0 = d01m;


#atomic ensemble properties

freq = np.arange(-15, 20.1, 0.1)*gd
step = lambd0
nat = 50 #number of atoms
nb = 3 #number of neighbours in raman chanel
displacement = 0.5*2*np.pi#1.5

"""
______________________________________________________________________________
Executing program
______________________________________________________________________________
"""


if __name__ == '__main__':
    chi = ensemble(nat,  #number of atoms
                   nb,    #number of neighbours in raman chanel
                   'fiber', #Stands for cross-section calculation(vacuum) or 
                             #transition calculations (fiber) 
                   'nocorrchain',  #Stands for atom positioning : chain, nocorrchain and doublechain
                   dist = displacement, # sigma for displacement
                   d=1.5, # distance from fiber
                   l0=step, # mean distance between atoms
                   deltaP=freq, # array of freq.
                   typ = 'L', # L or V for Lambda and V atom resp.
                   dens=1. # mean density for x-section calculations
                   )
               
    chi.generate_ensemble()
    chi.visualize() # use for refl and trans visualisation
    #chi.vis_crsec() #use for xsec. visualisation