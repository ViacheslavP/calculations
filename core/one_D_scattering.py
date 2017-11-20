
# -*- coding: utf-8 -*-

"""
Created 23.07.16
AS Sheremet's matlab code implementation 
n = 1.45 - the case that we only need to calculate

TODO: 


"""
try:
    import mkl
    mkl.set_num_threads(mkl.get_max_threads())
    mkl.service.set_num_threads(mkl.get_max_threads())
    
except ImportError:
    print('MKL does not installed. MKL dramatically improves calculation time')
    
    
from scipy.sparse import linalg as alg
from scipy.sparse import csc_matrix as csc


import numpy as np
from time import time

   

def inverse(ResInv,ddRight):
    return alg.spsolve(csc(ResInv),(ddRight))
    #return cu(csc(ResInv),csc(ddRight)).toarray()

class ensemble(object):
    
        """
        Class of atom ensemble
        """
        def __init__(self,
                     nat=5,
                     nb=4,
                     s='chain',
                     dist=0.,
                     d = 1.5,
                     l0=2*np.pi,
                     deltaP=np.asarray([0.]),
                     typ='L',
                     dens=0.1,
                     ff = 0.1
                     ):
                         
            self.rr = np.array([[]])
            self.ep = np.array([],dtype=np.complex)
            self.em = np.array([],dtype=np.complex)
            self.ez = np.array([],dtype=np.complex)
            self.ezm = np.array([],dtype=np.complex)
            self.epm = np.array([],dtype=np.complex)
            self.xm = np.zeros([nat,nat],dtype=np.complex)
            self.xp = np.array([[]],dtype=np.complex)
            self.x0 = np.array([[]],dtype=float)
            self.index = np.array([[]])
            self.Transmittance = np.zeros(len(deltaP),dtype=np.complex);
            self.Reflection = np.zeros(len(deltaP),dtype=np.complex);
            self.iTransmittance = np.zeros(len(deltaP),dtype=np.complex);
            self.iReflection = np.zeros(len(deltaP),dtype=np.complex);
            self.SideScattering = np.zeros(len(deltaP),dtype=float);
            self.CrSec = np.zeros(len(deltaP),dtype=np.complex)
            self._foo = [[[]]]
            self.g = 1.06586/c;  #decay rate corrected in presence of a nanofiber, units of gamma 
            
            from bmode import exact_mode
            m = exact_mode(1/lambd, n0, a)
            m.generate_mode()
            self.vg = m.vg*c #group velocity
            self.kv = m.k
            self.E = m.E
            self.dE = m.dE
            self.Ez = m.Ez
            #m.generate_sub_mode(d*a)
            self.kv_sub = 1. #m.sub_k
            
            self._dist = dist
            self._s = s
            self.d = d
            self.nat = nat
            self.nb = nb
            self.deltaP = deltaP
            self.step = l0 * np.pi / self.kv
            self.dens = dens
            self.typ = typ
            self.ff = ff
            self.r = 1
            self.L = 2 * np.pi / self.kv
            
        def generate_ensemble(self,dspl = True):
    
            """
            Method generates enseble for current atomic ensemble properties and
            finds requested number of neighbours
            
            """  
            step = self.step
            nat=self.nat
            s = self._s
            dist = self._dist
            nb = self.nb
            if FIX_RANDOM == 1:
                np.random.seed(seed=5)
            

            if s=='chain':
                x = self.d*a*np.ones(nat)
                y = 0.*np.ones(nat)
                z = np.arange(nat)*step
            elif s=='nocorrchain':
                x = self.d*a*np.ones(nat)
                y = 0.*np.ones(nat)
                L = (nat-1)*self.step
                z = np.random.rand(nat)*L
                z = np.sort(z)


            elif s=='doublechain':
                x = self.d*a*np.asarray([np.sign(i - nat/2) for i in range(nat)])
                y = 0.*np.ones(nat)
                goo = [i for i in range(nat//2)]
                z = np.asarray(goo+goo)*step
            
            
            elif s=='ff_chain':
                if self.ff>0.99:
                    raise ValueError
                
                N = int( nat / self.ff )
                L = N*self.step
                """
                
                Creates a chain with a filling factor = ff 
                I think that is not possible to have a two atoms in same location
                within this alghoritm 
                
                """
                x = self.d*a*np.ones(nat)
                y = 0.*np.ones(nat)
                z = (np.arange(nat)+np.sort(np.random.randint(N-nat,size=nat)))*step
            elif s == '45':
                x = self.d*a*np.ones(nat) /np.sqrt(2)
                y = -self.d*a*np.ones(nat) / np.sqrt(2)
                z = np.arange(nat)*step
            else:
                raise NameError('Ensemble type not found')
            
            if dist != .0 :
                pass
                x+= np.random.normal(0.0, dist*lambd, nat)
                y+= np.random.normal(0.0, dist*lambd, nat)
                z+= np.random.normal(0.0, dist*lambd, nat)
            
            

            t1 = time()
            

                
            """
            
            Exact mode components
            e stands for exact mode
            m,p,z stands for component
                
                
            minus - polarization 
            REM: def of polarization = irreducible representation of U(1) group (fumndamental mode only!) 
                
            """
            self.r = np.sqrt(x**2+y**2)
            
            self.em = -self.E(self.r)
                
            self.ep = PARAXIAL*-self.dE(self.r)*(x-1j*y)**2/self.r/self.r
            self.ez = PARAXIAL*self.Ez(self.r)*(x-1j*y)/self.r
                
            
            self.rr = np.asarray(np.sqrt([[((x[i]-x[j])**2+(y[i]-y[j])**2+(z[i] \
            -z[j])**2)for i in range(nat)] for j in range(nat)]),dtype=float)


            self.xm = np.asarray([[((x[i]-x[j])-1j*(y[i]-y[j]))/(self.rr[i,j] + \
            np.identity(nat)[i,j]) for j in range(nat)]for i in range(nat)],dtype=np.complex)*(1/np.sqrt(2))
        
            self.xp = -np.asarray([[((x[i]-x[j])+1j*(y[i]-y[j]))/(self.rr[i,j]+ \
            np.identity(nat)[i,j]) for j in range(nat)]for i in range(nat)],dtype=np.complex)*(1/np.sqrt(2))
        
            self.x0 = np.asarray([[(z[i]-z[j])/(self.rr[i,j] + \
            np.identity(nat)[i,j]) for j in range(nat)] for i in range(nat)],dtype=float)
            
            self.index = np.asarray(np.argsort(self.rr)[:,1:nb+1], dtype=np.int)
            
            #FREE_MEM
            import gc
            gc.collect()
                        
            self.create_D_matrix()
            self.reflection_calc()

            print('Current ensemble calculation time is ',time()-t1,'s')
 

        def create_D_matrix(self):
            """
            Method creates D matrix and reshapes D
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

            The Radiation Modes Model is just considering that radiation modes of fiber is vacuum modes
             minus guided modes that propagates with speed of light and vacuum wavelength
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

            
            
            """
            Selecting neighbours only (experimental idea for subtractions)
            """


            def foo2(na):
                if na > self.L:
                    return 0.
                else:
                    return 1.
            neigh = np.zeros([nat,nat])

            for i in range(nat):
                for j in range(nat):
                    neigh[i,j] = foo2(self.rr[i,j])

            kv = 1
            
            
            
            D1 = RADIATION_MODES_MODEL*kv*((1 - 1j*self.rr*kv - (self.rr*kv)**2)/ \
                ((self.rr*kv + np.identity(nat))**3) \
                *np.exp(1j*self.rr*kv)) *(np.ones(nat)-np.identity(nat))   
            D2 = RADIATION_MODES_MODEL*-1*hbar*kv*((3 - 3*1j*self.rr*kv - (self.rr*kv)**2)/((((kv*self.rr+\
            np.identity(nat))**3))*np.exp(1j*kv*self.rr)))
            
            Dz = np.zeros([nat,nat], dtype=np.complex)
            DzSub = np.zeros([nat,nat], dtype=np.complex)
            
            for i in range(nat):
                for j in range(nat):
    
                    Dz[i,j] =   \
                                -1*2j*np.pi*kv*np.exp(1j*self.kv*(abs(self.x0[i,j]))* \
                    self.rr[i,j])*(1/self.vg)
                    DzSub[i,j] = -1*2j*np.pi*kv*np.exp(1j*self.kv_sub*(abs(self.x0[i,j]))* \
                    self.rr[i,j])*(RADIATION_MODES_MODEL*neigh[i,j] / self.kv_sub)


            #Interaction part (and self-interaction for V-atom)
            Di = np.zeros([nat,nat,3,3], dtype = np.complex)


            for i in range(nat):
                for j in range(nat):
                    
                    if (i==j) and (self.typ == 'L'):
                        continue
                    
                    """
                    ________________________________________________________
                    Guided modes interaction (and self-energy for V atom)
                    ________________________________________________________
                    
                    Using next notations:
                        epfjc - vector guided mode of Electric field 
                                                   in Plus polarisation (c REM in 176)
                                          propagating Forward
                                                   in J-th atom position
                                                  and Conjugated components
                    
                    forward - part of Green's function for propagating forward (!)
                    backward - guess what
                    
                    The Result for self-energy is obtained analytically
                    
                    
                                                       e+            e0           e-
                      
                    """

                    #Full modes - exact solutions

                    emfi  =  np.conjugate(  np.array([self.ep[i], self.ez[i], self.em[i]], dtype=np.complex))
                    emfjc =                 np.array([self.ep[j], self.ez[j], self.em[j]], dtype=np.complex)


                    epfi  =                 np.array([self.em[i], self.ez[i], self.ep[i]], dtype=np.complex)
                    epfjc =  np.conjugate(  np.array([self.em[j], self.ez[j], self.ep[j]], dtype=np.complex) )
                    
                    
                    
                    embi  =  np.conjugate(  np.array([-self.ep[i], self.ez[i], -self.em[i]], dtype=np.complex))
                    embjc =                 np.array([-self.ep[j], self.ez[j], -self.em[j]], dtype=np.complex)
                    
                    epbi  =                 np.array([-self.em[i], self.ez[i], -self.ep[i]], dtype=np.complex)
                    epbjc =  np.conjugate(  np.array([-self.em[j], self.ez[j], -self.ep[j]], dtype=np.complex) )

                    #Perp mode - projected onto polarization vectors

                    emfjcSub = np.array([0, 0, self.em[j]], dtype=np.complex)
                    epfjcSub = np.conjugate(np.array([self.em[j], 0, 0], dtype=np.complex))

                    embjcSub = np.array([0, 0, -self.em[j]], dtype=np.complex)
                    epbjcSub = np.conjugate(np.array([-self.em[j], 0, 0], dtype=np.complex))

                    """
                    Forward and Backward (and its sub-duality!) matrices are symmetric. 
                    Moreover: they're cylindricaly symmetric!
                    """
                    forwardSub = np.outer(emfi, emfjcSub) + np.outer(epfi, epfjcSub)
                    backwardSub = np.outer(embi, embjcSub) + np.outer(epbi, epbjcSub)

                    forward = np.outer(emfi, emfjc) + np.outer(epfi, epfjc)
                    backward = np.outer(embi, embjc) + np.outer(epbi, epbjc)


                    zi = float(self.x0[i,0]*self.rr[i,0])
                    zj = float(self.x0[j,0]*self.rr[j,0])


                    if abs(zi - zj) < 1e-6:
                        Di[i,j,:,:] = 0.5 * d00*d00*((forward + backward) * Dz[i,i] - (forwardSub+backwardSub)*DzSub[i,j])
                        
                    elif zi<zj:
                        Di[i,j,:,:] = (forward * Dz[i,j] - forwardSub*DzSub[i,j]) * d00*d00
                        
                    elif zi>zj:
                        Di[i,j,:,:] = (backward * Dz[i,j] - backwardSub*DzSub[i,j]) * d00*d00


                    """
                    _________________________________________________________
                    +Vacuum interaction (No self-energy)
                    _________________________________________________________
                    """
                        
                    Di[i,j,0,0] += d01m*d1m0*(D1[i,j] -\
                                            self.xp[i,j]*self.xm[i,j]*D2[i,j])
                                            
                    
                    Di[i,j,0,1] += d01m*d00*(-self.xp[i,j]*self.x0[i,j]*D2[i,j]);
                                            
                    Di[i,j,0,2] += d01m*d10*(-self.xp[i,j]*self.xp[i,j]*D2[i,j] );
                                            
                    Di[i,j,1,0] += d00*d1m0*(self.x0[i,j]*self.xm[i,j]*D2[i,j] );
                                            
                    Di[i,j,1,1] += d00*d00*(D1[i,j] + \
                                           self.x0[i,j]*self.x0[i,j]*D2[i,j]);
                                           
                    Di[i,j,1,2] += d00*d10*(self.x0[i,j]*self.xp[i,j]*D2[i,j] );
                                            
                    Di[i,j,2,0] += d01*d1m0*(-self.xm[i,j]*self.xm[i,j]*D2[i,j]);
                                            
                    Di[i,j,2,1] += d01*d00*(-self.xm[i,j]*self.x0[i,j]*D2[i,j]);
                                            
                    Di[i,j,2,2] += d01*d10*(D1[i,j]- \
                                            self.xm[i,j]*self.xp[i,j]*D2[i,j])


            if self.typ == 'L':
                try:

                    from sigmaMatrix import returnForLambda
                    self.D = returnForLambda(Di, np.array(self.index, dtype=np.int), nb)
                except:
                    print("sigmaMatrix.pyx not compiled! Using pure python to form Simgma-matrix.")

                    from itertools import product as combi
                    state = np.asarray([i for i in combi(range(3),repeat=nb)])
                    st = np.ones([nat,nat,3**nb], dtype = int)*2
                    for n1 in range(nat):
                        k=0
                        for n2 in np.sort(self.index[n1,:]):
                            for i in range(3**nb):
                                st[n1,n2,i] = state[i,k]
                            k+=1

                    self.D = np.zeros([3**nb*nat,3**nb*nat],dtype=np.complex)

                    for n1 in range(nat):
                        for n2 in range(nat):#choose initial excited atom
                            for i in range(3**nb):  #select initial excited atom environment
                            #choose final excited or np.sort(self.index[n1,:]): if massive photon
                              
                                for j in range(3**nb):      #select final excited environment
                                    if foo(n1,n2,i,j):  #if transition is possible then make assigment
                                        self.D[n1*3**nb+i,n2*3**nb+j] =  \
                                        Di[n1,n2,st[n1,n2,i],st[n2,n1,j]]

            elif self.typ == 'V':

                try:
                    from sigmaMatrix import returnForV
                    self.D = returnForV(Di)

                except:
                    print("sigmaMatrix.pyx not compiled! Using pure python to form Simgma-matrix.")

                    self.D = np.zeros([3*nat,3*nat],dtype=np.complex)
                    for n1 in range(nat): #initial excited
                    
                        for n2 in range(nat):#final excited
                            for i in range(3):
                        
                        
                                for j in range(3):
                                    self.D[3*n1+j,3*n2+i] = 3*Di[n1,n2,j,i]

            else:
                raise NameError('No such type')
            
        
        def S_matrix_for_V(self):
            import sys
            nat=self.nat
            nb = self.nb
            k=1
            kv=1
            c=1
            
            """
            ________________________________________________________________
            Definition of channels of Scattering
            ________________________________________________________________
            """
            
            ddLeftF = np.zeros(3*nat, dtype=np.complex);
            ddLeftB = np.zeros(3*nat, dtype=np.complex);
            ddLeftFm = np.zeros(3*nat, dtype=np.complex);
            ddLeftBm = np.zeros(nat*3, dtype=np.complex);
            ddRight = np.zeros(3*nat, dtype=np.complex); 
            One = (np.identity(3*nat)) #Unit in V-atom
            Sigmad = (np.identity(nat*3,dtype=np.complex))
            for i in range(nat):

                #--In-- chanel
                ddRight[3*i] = np.sqrt(3)*1j*d10*np.exp(-1j*self.kv*self.x0[0,i]*self.rr[0,i])  \
                    *self.em[i]
                ddRight[3*i+1] = np.sqrt(3)*1j*d10*np.exp(-1j*self.kv*self.x0[0,i]*self.rr[0,i])  \
                    *self.ez[i] 
                ddRight[3*i+2] = np.sqrt(3)*1j*d10*np.exp(-1j*self.kv*self.x0[0,i]*self.rr[0,i])  \
                    *self.ep[i]
                    
                #-- Out Forward --
                ddLeftF[3*i] = np.sqrt(3)*-1j*d01*np.exp(+1j*self.kv*self.x0[0,i]*self.rr[0,i]) \
                    *np.conjugate(self.em[i])
                ddLeftF[3*i+1] = np.sqrt(3)*-1j*d01*np.exp(+1j*self.kv*self.x0[0,i]*self.rr[0,i]) \
                    *np.conjugate(self.ez[i])
                ddLeftF[3*i+2] = np.sqrt(3)*-1j*d01*np.exp(+1j*self.kv*self.x0[0,i]*self.rr[0,i]) \
                    *np.conjugate(self.ep[i])
                    
                # -- Out Backward --
                ddLeftB[3*i] = -np.sqrt(3)*-1j*d01*np.exp(-1j*self.kv*self.x0[0,i]*self.rr[0,i]) \
                    *np.conjugate(self.em[i])
                ddLeftB[3*i+1] = np.sqrt(3)*-1j*d01*np.exp(-1j*self.kv*self.x0[0,i]*self.rr[0,i]) \
                    *np.conjugate(self.ez[i])
                ddLeftB[3*i+2] = -np.sqrt(3)*-1j*d01*np.exp(-1j*self.kv*self.x0[0,i]*self.rr[0,i]) \
                    *np.conjugate(self.ep[i])
                    
                    
                #-- Out Forward -- (inelastic)
                ddLeftFm[3*i] = np.sqrt(3)*-1j*d01*np.exp(+1j*self.kv*self.x0[0,i]*self.rr[0,i]) \
                    *(self.ep[i])
                ddLeftFm[3*i+1] = np.sqrt(3)*-1j*d01*np.exp(+1j*self.kv*self.x0[0,i]*self.rr[0,i]) \
                    *(self.ez[i])
                ddLeftFm[3*i+2] = np.sqrt(3)*-1j*d01*np.exp(+1j*self.kv*self.x0[0,i]*self.rr[0,i]) \
                    *(self.em[i])
                    
                # -- Out Backward -- (inelastic)
                ddLeftBm[3*i] = -np.sqrt(3)*-1j*d01*np.exp(-1j*self.kv*self.x0[0,i]*self.rr[0,i]) \
                    *(self.ep[i])
                ddLeftBm[3*i+1] = np.sqrt(3)*-1j*d01*np.exp(-1j*self.kv*self.x0[0,i]*self.rr[0,i]) \
                    *(self.ez[i])
                ddLeftBm[3*i+2] = -np.sqrt(3)*-1j*d01*np.exp(-1j*self.kv*self.x0[0,i]*self.rr[0,i]) \
                    *(self.em[i])
                    
           
                for j in range(3):
                    Sigmad[(i)*3+j,(i)*3+j] = VACUUM_DECAY*0.5j  #Vacuum decay

            """
            ________________________________________________________________
            Calculation of Scattering matrix elements

            ________________________________________________________________
            """
            for k in range(len(self.deltaP)):
                
                omega = self.deltaP[k]
                    
                Sigma = Sigmad+omega*One
                V =  1*(- self.D/hbar/lambd/lambd )
                ResInv = Sigma + V   
                Resolventa =  np.linalg.solve(ResInv,ddRight)
                TF_ = np.dot(Resolventa,ddLeftF)*2*np.pi*hbar*kv
                TB_ = np.dot(Resolventa,ddLeftB)*2*np.pi*hbar*kv
                iTF_ = np.dot(Resolventa,ddLeftFm)*2*np.pi*hbar*kv
                iTB_ = np.dot(Resolventa,ddLeftBm)*2*np.pi*hbar*kv
                    
                self.Transmittance[k] = 1.+(-1j/hbar*(TF_)/(self.vg))
                self.Reflection[k] = (-1j/hbar*(TB_/(self.vg)))
                    
                    
                self.iTransmittance[k] = (-1j/hbar*(iTF_)/(self.vg))                  
                self.iReflection[k] =  (-1j/hbar*(iTB_/(self.vg)))
                     
                self.SideScattering[k] =1 - abs(self.Transmittance[k])**2-abs(self.Reflection[k])**2-\
                                          abs(self.iTransmittance[k])**2-abs(self.iReflection[k])**2
                                          
                ist = 100*k / len(self.deltaP)
                sys.stdout.write("\r%d%%" % ist)
                sys.stdout.flush()
                
                

            print('\n')


        def S_matrix_for_Lambda(self):

            """
            Method calculates S matrix for Lambda atom. There is two global options, which is set by changing global
            parameter SINGLE_RAMAN to 1 or 0. For 0 it calculates S matrix elements only for Rayleigh and Faraday
            channels (without change of atomic polarization). For 1 it calculates these ... for changing polarization of
            single atom in a whole atomic chain.

            :return:
            fullTransmittance and fullReflection
            """
            import sys


            nat=self.nat
            nb = self.nb
            k=1
            kv=1
            c=1
            
            #Decay rate for Lambda atom with respect of guided modes
            gd = VACUUM_DECAY*1+8*d00*d00*np.pi*k*((1/self.vg - RADIATION_MODES_MODEL/c)*(abs(self.em)**2 + abs(self.ep)**2 + \
                                                    abs(self.ez)**2)  + FIRST*RADIATION_MODES_MODEL/c * abs(self.ez)**2 \
                                                                      + SECOND*RADIATION_MODES_MODEL/c * abs(self.ep)**2 )


            """
            ________________________________________________________________
            Definition of channels of Scattering
            ________________________________________________________________
            """
            #Input
            ddRight = np.zeros(nat*3**nb, dtype=np.complex);  


            """
            ________________________________________________________________
            Rayleigh channels (m = +1 -> m'= +1, p = +1 -> p' = +1)
            ________________________________________________________________
            """
            #Output
                
            #+1, -
            ddLeftF_pm = np.zeros(nat*3**nb, dtype=np.complex);
            ddLeftB_pm = np.zeros(nat*3**nb, dtype=np.complex);

            # +1, + (smwht like Faraday effect of magnitized atomic chain(!), no atomic transitions, will call it Faraday channel)
            ddLeftF_pp = np.zeros(nat * 3 ** nb, dtype=np.complex);
            ddLeftB_pp = np.zeros(nat * 3 ** nb, dtype=np.complex);

            if SINGLE_RAMAN:

                """
                ________________________________________________________________
                Raman channels
                ________________________________________________________________
                """



                #0, +
                ddLeftF_0p = np.zeros([nat*3**nb, nat], dtype=np.complex);
                ddLeftB_0p = np.zeros([nat*3**nb, nat], dtype=np.complex);
                
                #0, -
                ddLeftF_0m = np.zeros([nat*3**nb, nat], dtype=np.complex);
                ddLeftB_0m = np.zeros([nat*3**nb, nat], dtype=np.complex);
                
                #-1, +
                ddLeftF_mp = np.zeros([nat*3**nb, nat], dtype=np.complex);
                ddLeftB_mp = np.zeros([nat*3**nb, nat], dtype=np.complex);
                
                #-1, -
                ddLeftF_mm = np.zeros([nat*3**nb, nat], dtype=np.complex);
                ddLeftB_mm = np.zeros([nat*3**nb, nat], dtype=np.complex);

                # Reduction of T-Raman-Matrix
                sum_reduce = lambda A: np.add.reduce(np.square(np.absolute(A)))

            One = (np.identity(nat*3**nb)) # Unit in Lambda-atom with nb neighbours
            Sigmad = 0*(np.identity(nat*3**nb,dtype=np.complex))

            #Loop over atoms, from which the photon emits
            for i in range(nat):


                    
                ddRight[(i+1)*3**nb-1] = 1j*d10*np.exp(-1j*self.kv*self.x0[0,i]*self.rr[0,i])*self.em[i]

                """
                ________________________________________________________________
                Rayleigh channels (m = +1 -> m'= +1, p = +1 -> p' = +1)
                ________________________________________________________________
                """
                # Rayleigh
                ddLeftF_pm[(i+1)*3**nb-1] = -1j*d01*np.exp(+1j*self.kv*self.x0[0,i]*self.rr[0,i])*np.conjugate(self.em[i])
                ddLeftB_pm[(i+1)*3**nb-1] = +1j*d01*np.exp(-1j*self.kv*self.x0[0,i]*self.rr[0,i])*np.conjugate(self.em[i])
                # Faraday
                ddLeftF_pp[(i+1)*3**nb-1] = -1j * d01*np.exp(+1j * self.kv * self.x0[0, i] * self.rr[0, i]) * self.ep[i]
                ddLeftB_pp[(i+1)*3**nb-1] =  1j * d01*np.exp(-1j * self.kv * self.x0[0, i] * self.rr[0, i]) * self.ep[i]


                if SINGLE_RAMAN:
                    """
                    ________________________________________________________________
                    Raman channels
                    ________________________________________________________________
                    """
                    #loop over atoms, which change their pseudospin projection
                    #for jp in range(nb+1):
                    counter = 0
                    for jp in np.append(np.asarray([i],dtype=np.int),self.index[i,:]):

                        if jp!=i:
                            jc=counter
                        elif jp==i:
                            index_0 = (i+1)*3**nb - 1
                            index_m = (i+1)*3**nb - 1
                            j = i

                            ddLeftF_0m[index_0, jp] = -1j * d01 * np.exp(+1j * self.kv * self.x0[0, i] * self.rr[0, i]) * np.conjugate(self.ez[i])
                            ddLeftB_0m[index_0, jp] = -1j * d01 * np.exp(-1j * self.kv * self.x0[0, i] * self.rr[0, i]) * np.conjugate(self.ez[i])

                            ddLeftF_0p[index_0, jp] = -1j * d01 * np.exp(+1j * self.kv * self.x0[0, i] * self.rr[0, i]) * \
                                                     self.ez[i]
                            ddLeftB_0p[index_0, jp] = -1j * d01 * np.exp(-1j * self.kv * self.x0[0, i] * self.rr[0, i]) * \
                                                     self.ez[i]

                            ddLeftF_mm[index_m, jp] = -1j * d01 * np.exp(+1j * self.kv * self.x0[0, i] * self.rr[0, i]) * np.conjugate(self.ep[i])
                            ddLeftB_mm[index_m, jp] = 1j * d01 * np.exp(-1j * self.kv * self.x0[0, i] * self.rr[0, i]) * np.conjugate(self.ep[i])

                            ddLeftF_mp[index_m, jp] = -1j * d01 * np.exp(+1j * self.kv * self.x0[0, i] * self.rr[0, i]) * \
                                                     self.em[i]
                            ddLeftB_mp[index_m, jp] = 1j * d01 * np.exp(-1j * self.kv * self.x0[0, i] * self.rr[0, i]) * \
                                                     self.em[i]

                            continue

                        counter += 1
                        index_0 = (i)*3**nb - 1 + 3**nb - (2-1)*(3**(jc))
                        index_m = (i)*3**nb - 1 + 3**nb - (2-0)*(3**(jc))

                        ddLeftF_0m[index_0,jp] = -1j*d01*np.exp(+1j*self.kv*self.x0[0,i]*self.rr[0,i])*np.conjugate(self.em[i])
                        ddLeftB_0m[index_0,jp] =  1j*d01*np.exp(-1j*self.kv*self.x0[0,i]*self.rr[0,i])*np.conjugate(self.em[i])
                    
                        ddLeftF_0p[index_0,jp] = -1j*d01*np.exp(+1j*self.kv*self.x0[0,i]*self.rr[0,i])*self.ep[i]
                        ddLeftB_0p[index_0,jp] =  1j*d01*np.exp(-1j*self.kv*self.x0[0,i]*self.rr[0,i])*self.ep[i]
                    
                        ddLeftF_mm[index_m,jp] = -1j*d01*np.exp(+1j*self.kv*self.x0[0,i]*self.rr[0,i])*np.conjugate(self.em[i])
                        ddLeftB_mm[index_m,jp] =  1j*d01*np.exp(-1j*self.kv*self.x0[0,i]*self.rr[0,i])*np.conjugate(self.em[i])
                    
                        ddLeftF_mp[index_m,jp] = -1j*d01*np.exp(+1j*self.kv*self.x0[0,i]*self.rr[0,i])*self.ep[i]
                        ddLeftB_mp[index_m,jp] =  1j*d01*np.exp(-1j*self.kv*self.x0[0,i]*self.rr[0,i])*self.ep[i]
                        """
                        if j==i:
                            index_0 = (i+1)*3**nb - 1
                            index_m = (i+1)*3**nb - 1

                            ddLeftF_0m[index_0, j] = -1j * d01 * np.exp(+1j * self.kv * self.x0[0, i] * self.rr[0, i]) * np.conjugate(self.ez[i])
                            ddLeftB_0m[index_0, j] = -1j * d01 * np.exp(-1j * self.kv * self.x0[0, i] * self.rr[0, i]) * np.conjugate(self.ez[i])

                            ddLeftF_0p[index_0, j] = -1j * d01 * np.exp(+1j * self.kv * self.x0[0, i] * self.rr[0, i]) * \
                                                     self.ez[i]
                            ddLeftB_0p[index_0, j] = -1j * d01 * np.exp(-1j * self.kv * self.x0[0, i] * self.rr[0, i]) * \
                                                     self.ez[i]

                            ddLeftF_mm[index_m, j] = -1j * d01 * np.exp(+1j * self.kv * self.x0[0, i] * self.rr[0, i]) * np.conjugate(self.ep[i])
                            ddLeftB_mm[index_m, j] =  1j * d01 * np.exp(-1j * self.kv * self.x0[0, i] * self.rr[0, i]) * np.conjugate(self.ep[i])

                            ddLeftF_mp[index_m, j] = -1j * d01 * np.exp(+1j * self.kv * self.x0[0, i] * self.rr[0, i]) * \
                                                     self.em[i]
                            ddLeftB_mp[index_m, j] = 1j * d01 * np.exp(-1j * self.kv * self.x0[0, i] * self.rr[0, i]) * \
                                                     self.em[i]
                        """
                    
                for j in range(3**nb):
                    Sigmad[(i)*3**nb+j,(i)*3**nb+j] = gd[i]*0.5j #distance dependance of dacay rate

            self.fullTransmittance = np.zeros(len(self.deltaP), dtype=float)
            self.fullReflection = np.zeros(len(self.deltaP), dtype=float)

            for k in range(len(self.deltaP)):
                
                omega = self.deltaP[k]
                 
                #V = lm(np.zeros([nat*3**nb,nat*3**nb], dtype = np.complex))
                Sigma = (Sigmad+(omega- RABI**2 / (4* (omega - DC + 1e-6j) ))*One)
                V =  1*(- self.D/hbar/lambd/lambd )

                ResInv = Sigma + V

                
                Resolventa = np.linalg.solve(ResInv,ddRight)
                #Resolventa = __solver(ResInv, ddRight)

                TF_pm = np.dot(Resolventa,ddLeftF_pm)*2*np.pi*hbar*kv
                TB_pm = np.dot(Resolventa,ddLeftB_pm)*2*np.pi*hbar*kv

                TF_pp = np.dot(Resolventa, ddLeftF_pp) * 2 * np.pi * hbar * kv
                TB_pp = np.dot(Resolventa, ddLeftB_pp) * 2 * np.pi * hbar * kv

                self.Transmittance[k] = 1.+(-1j/hbar*(TF_pm)/(self.vg))
                self.Reflection[k] = (-1j/hbar*(TB_pm/(self.vg)))

                self.iTransmittance[k] = (-1j/hbar*(TF_pp)/(self.vg))
                self.iReflection[k] = (-1j/hbar*(TB_pp)/(self.vg))

                     
                self.SideScattering[k] = 1 - abs(self.Transmittance[k])**2 - abs(self.Reflection[k])**2 - \
                                         abs(self.iTransmittance[k])**2 - abs(self.iReflection[k])**2

                self.fullTransmittance[k] =  abs(self.Transmittance[k]) ** 2 + \
                                            abs(self.iTransmittance[k]) ** 2

                self.fullReflection[k] = abs(self.Reflection[k]) ** 2 + \
                                         abs(self.iReflection[k]) ** 2


                if SINGLE_RAMAN: #SINGLE_RAMAN == 1:




                    TF2_0m = sum_reduce(np.dot(Resolventa, ddLeftF_0m) * 2 * np.pi * hbar * kv)
                    TB2_0m = sum_reduce(np.dot(Resolventa, ddLeftB_0m) * 2 * np.pi * hbar * kv)

                    TF2_0p = sum_reduce(np.dot(Resolventa, ddLeftF_0p) * 2 * np.pi * hbar * kv)
                    TB2_0p = sum_reduce(np.dot(Resolventa, ddLeftB_0p) * 2 * np.pi * hbar * kv)

                    TF2_mm = sum_reduce(np.dot(Resolventa, ddLeftF_mm) * 2 * np.pi * hbar * kv)
                    TB2_mm = sum_reduce(np.dot(Resolventa, ddLeftB_mm) * 2 * np.pi * hbar * kv)

                    TF2_mp = sum_reduce(np.dot(Resolventa, ddLeftF_mp) * 2 * np.pi * hbar * kv)
                    TB2_mp = sum_reduce(np.dot(Resolventa, ddLeftB_mp) * 2 * np.pi * hbar * kv)



                    self.fullTransmittance[k] +=((TF2_0m) + \
                                                (TF2_0p) + \
                                                (TF2_mm) +  \
                                                (TF2_mp) ) *(1/hbar/self.vg)**2

                    self.fullReflection[k] +=((TB2_0m) + \
                                             (TB2_0p) + \
                                             (TB2_mm) + \
                                             (TB2_mp) ) *(1/hbar/self.vg)**2

                    self.SideScattering[k] = 1 - (self.fullTransmittance[k]) - (self.fullReflection[k])
                                          
                ist = 100*k / len(self.deltaP)
                sys.stdout.write("\r%d%%" % ist)
                sys.stdout.flush()
                
            print('\n')


        def reflection_calc(self):
            if self.typ == 'V':
                self.S_matrix_for_V()
            elif self.typ == 'L':
                self.S_matrix_for_Lambda()
            else:
                raise NameError('Wrong type of atom')
                
            
        def visualize(self,addit='',save=True):
            import matplotlib.pyplot as plt
            
            plt.plot(self.deltaP,(abs(self.Reflection)**2 ), 'r-',label='R',lw=1.5)
            plt.plot(self.deltaP, abs(self.Transmittance)**2, 'm-', label='T',lw=1.5)
            plt.legend()
            plt.xlabel('Detuning, $\gamma$',fontsize=16)
            plt.ylabel('R,T',fontsize=16)
            plt.savefig('TnR.png',dpi=700)
            plt.show()
            plt.clf()

            #plt.plot(self.deltaP, np.gradient(np.angle(self.Reflection) ), 'r-',label='arg R',lw=1.5)
            plt.plot(self.deltaP, 2*np.pi*np.gradient(np.angle(self.Transmittance)), 'm-', label='arg T',lw=1.5)
            plt.legend()
            plt.xlabel('Detuning, $\gamma$',fontsize=16)
            plt.ylabel('R,T',fontsize=16)
            plt.savefig('TnR.png',dpi=700)
            plt.show()
            plt.clf()

            plt.plot(self.deltaP,(abs(self.iReflection)**2 ), 'r-',label='R',lw=1.5)
            plt.plot(self.deltaP, abs(self.iTransmittance)**2, 'm-', label='T',lw=1.5)
            plt.legend()
            plt.xlabel('Detuning, $\gamma$',fontsize=16)
            plt.ylabel('R,T',fontsize=16)
            plt.savefig('TnR.png',dpi=700)
            plt.show()
            plt.clf()

            if self.typ == 'V':
                plt.plot(self.deltaP,(abs(self.iReflection)**2 ), 'r-',label='R',lw=1.5)
                plt.plot(self.deltaP, abs(self.iTransmittance)**2, 'm-', label='T',lw=1.5)
                plt.legend()
                plt.xlabel('Detuning, $\gamma$',fontsize=16)
                plt.ylabel('R,T',fontsize=16)
                plt.savefig('TnR_rot.png',dpi=700)
                plt.show()
                plt.clf()

            if self.typ == 'L' and SINGLE_RAMAN==True:
                plt.plot(self.deltaP,self.fullReflection, 'r-',label='R',lw=1.5)
                plt.plot(self.deltaP, self.fullTransmittance, 'm-', label='T',lw=1.5)
                plt.legend()
                plt.xlabel('Detuning, $\gamma$',fontsize=16)
                plt.ylabel('R,T',fontsize=16)
                plt.savefig('TnR_full.png',dpi=700)
                plt.show()
                plt.clf()

            
            plt.xlabel('Detuning, $\gamma$')
            plt.ylabel('Loss')
            plt.title('Loss')
            plt.plot(self.deltaP, self.SideScattering, 'g-',lw=1.5)
            plt.show()
            plt.clf()
            
 #scale constants

        def simple_visor(self):
            import matplotlib.pyplot as plt

            plt.plot(self.deltaP, (abs(self.Reflection) ** 2), 'r-', label='R', lw=1.5)
            #plt.plot(self.deltaP, abs(self.Transmittance) ** 2, 'm-', label='T', lw=1.5)

            plt.plot(self.deltaP, abs(self.fullReflection) , 'r--', label='R', lw=1.5)
            #plt.plot(self.deltaP, abs(self.fullTransmittance) , 'm--', label='T', lw=1.5)
            plt.legend()
            plt.xlabel('Detuning, $\gamma$', fontsize=16)
            plt.ylabel('R,T', fontsize=16)
            plt.savefig('TnR.png', dpi=700)
            plt.show()
            plt.clf()


hbar = 1 #dirac's constant = 1 => enrergy = omega
c = 1 #speed of light c = 1 => time in cm => cm in wavelenght
gd = 1. #vacuum decay rate The only
lambd = 1 # atomic wavelength (lambdabar)
k = 1/lambd
a = 2*np.pi*200/780*lambd # nanofiber radius in units of Rb wavelength
n0 = 1.45 #(silica)



#dipole moments (Wigner-Eckart th.), atom properies

d01 = np.sqrt(hbar*gd/4*(lambd**3))  #|F = 0, n = 0> <- |F0 = 1> - populated level
d10 = d01
d00 = np.sqrt(hbar*gd/4*lambd**3)
d01m = np.sqrt(hbar*gd/4*lambd**3)
d1m0 = d01m


#atomic ensemble properties

freq = np.linspace(-2,2, 280)*gd

#Validation (all = 1 iff ful theory, except SINGLE_RAMAN)

RADIATION_MODES_MODEL = 0 # = 1 iff assuming our model of radiation modes =0 else
VACUUM_DECAY = 0# = 0 iff assuming only decay into fundamental mode, =1 iff decay into fundamental and into radiation
PARAXIAL = 1 # = 0 iff paraxial, =1 iff full mode
SINGLE_RAMAN = True
FIX_RANDOM = 1
RABI = 1e-16
DC = 0.

FULL = 0
FIRST = 1 # = 0 iff assuming full subtraction
SECOND = 1

"""
______________________________________________________________________________
Executing program
______________________________________________________________________________
"""



if __name__ == '__main__':

    #from matplotlib import rc

    #rc('font', **{'family': 'serif', 'sans-serif': ['Helvetica'], 'size': 16})
    #rc('text', usetex=True)

    args = {
        
            'nat':3, #number of atoms
            'nb':2, #number of neighbours in raman chanel (for L-atom only)
            's':'chain', #Stands for atom positioning : chain, nocorrchain and doublechain
            'dist':0.,  # sigma for displacement (choose 'chain' for gauss displacement.)
            'd' : 1.5, # distance from fiber
            'l0': 1.000, # mean distance between atoms (in lambda_m /2 units)
            'deltaP':freq,  # array of freq.
            'typ':'L',  # L or V for Lambda and V atom resp.
            'ff': 0.3
            }

    

    SE0 = ensemble(**args)
    SE0.L = 0.
    SE0.generate_ensemble()
    SE0.visualize()
    SE0.simple_visor()


