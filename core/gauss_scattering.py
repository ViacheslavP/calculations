
# -*- coding: utf-8 -*-

"""
Created 06.02.19
AS Sheremet's matlab code implementation

TODO: 


"""
    
    
from scipy.sparse import linalg as alg
from scipy.sparse import csc_matrix as csc

import numpy as np
from time import time


class ensemble(object):
    
        """
        Class of atom ensemble
        """
        def __init__(self,
                     nat=5,
                     nb=4,
                     s='chain',
                     dist=0.,
                     d=0,
                     l0=1.,
                     freqs=np.asarray([0.]),
                     typ='L',
                     dens=0.1,
                     ff = 0.1
                     ):
                         
            self.rr = np.array([[]])
            self.ep = np.array([],dtype=np.complex)
            self.xm = np.zeros([nat,nat],dtype=np.complex)
            self.xp = np.array([[]],dtype=np.complex)
            self.x0 = np.array([[]],dtype=float)
            self.index = np.array([[]])
            self.Transmittance = np.zeros(len(freqs), dtype=np.complex);
            self.Reflection = np.zeros(len(freqs), dtype=np.complex);
            self.iTransmittance = np.zeros(len(freqs), dtype=np.complex);
            self.iReflection = np.zeros(len(freqs), dtype=np.complex);
            self.SideScattering = np.zeros(len(freqs), dtype=float);
            self.CrSec = np.zeros(len(freqs), dtype=np.complex)
            
            from gmode import GaussMode
            m = GaussMode(WAIST, OMEGA)
            self.forward_mode = m.forward_amplitude
            self.backward_mode = m.backward_amplitude

            self._dist = dist
            self._s = s
            self.d = d
            self.nat = nat
            self.nb = nb
            self.freqs = freqs
            self.step = l0 * np.pi / (1 - 1/m.z0)
            self.dens = dens
            self.typ = typ
            self.ff = ff
            self.r = 1
            if PAIRS:
                self.nb = 0
            
        def generate_ensemble(self,dspl = True):
    
            """
            Method generates enseble for current atomic ensemble properties and
            finds requested number of neighbours
            
            """  
            step = self.step
            nat = self.nat
            s = self._s #Type of ensemble (random,chain,etc.)
            dist = self._dist #dist for distribution
            nb = self.nb
            if FIX_RANDOM == 1:
                np.random.seed(seed=SEED)

            if s == 'chain':
                x = self.d*np.ones(nat)                 # [d,d,d...]
                y = np.zeros(nat)                       # [0,0,0...]
                z = (np.arange(nat)-(nat-1.)/2.)*step   # [-(N-1)/2*step,...,(N-1)/2*step]


            elif s == 'nocorrchain':
                x = self.d*np.ones(nat)                 # [d,d,d...]
                y = 0.*np.ones(nat)
                L = (nat-1)*step
                z = np.random.rand(nat)*L
                z = np.sort(z)
                z0 = 1/nat * sum(z)
                z = z-z0                                # Centering
            elif s=='square':

                nr = np.int64(np.sqrt(nat))
                if np.int64(nat) != nr*nr:
                    raise ValueError
                x = np.zeros(nat)                 # [d,d,d...]
                y = np.zeros(nat)                       # [0,0,0...]
                z = np.zeros(nat)
                symfoo = (np.arange(nr) - (nr-1)/2.)*step
                for i in range(nr):
                    for j in range(nr):
                        x[i*nr + j] = symfoo[i]
                        y[i*nr + j] = symfoo[j]
                print(x,y)



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

            #Mirror-quench were just setups to prove that from left and from right our system
            #looks different for

            elif s=='mirror_quench':
                x = self.d * a * np.ones(nat)
                y = 0. * np.ones(nat)
                n1 = nat//2
                n2 = nat - n1
                foo = []
                for i in range(nat):
                    if i < n1:
                        foo.append(i)
                    else:
                        foo.append(n1+0.5*(i-n1))
                z = step*np.asarray(foo)



            elif s=='quench_mirror':
                x = self.d * a * np.ones(nat)
                y = 0. * np.ones(nat)
                n1 = nat//2
                n2 = nat - n1
                foo = []
                n1 = nat//2
                n2 = nat - n1
                foo = []
                for i in range(nat):
                    if i < n1:
                        foo.append(0.5*i)
                    else:
                        foo.append(i - 0.5*n1)
                z = step*np.asarray(foo)


            elif s == 'peculiar':
                from densityIter import rhoMCMC
                x = self.d*a*np.ones(nat)
                y = 0.*np.ones(nat)
                z = 2*step*rhoMCMC(nat)
                z = np.sort(z)

            else:
                raise NameError('Ensemble type not found')
            
            if dist != .0 :
                pass
                #x+= np.random.normal(0.0, dist*step, nat)
                #y+= np.random.normal(0.0, dist*step, nat)
                z+= np.random.normal(0.0, dist*step, nat)
            
            

            t1 = time()

            self.z = z
            self.r = np.sqrt(x ** 2 + y ** 2)

                
            """
            
            Mode components
        
            """

            self.e_forward = self.forward_mode(self.r, self.z)
            self.e_backward = self.backward_mode(self.r, self.z)

            if OPPOSITE_SCATTERING:
                self.e_forward = self.backward_mode(self.r, self.z)
                self.e_backward = self.forward_mode(self.r, self.z)

            self.rr = np.asarray(np.sqrt([[((x[i]-x[j])**2+(y[i]-y[j])**2+(z[i] \
            -z[j])**2)for i in range(nat)] for j in range(nat)]),dtype=float)

            self.xm = np.asarray([[((x[i]-x[j])-1j*(y[i]-y[j]))/(self.rr[i,j] + \
            np.identity(nat)[i,j]) for j in range(nat)]for i in range(nat)],dtype=np.complex)*(1/np.sqrt(2))
        
            self.xp = -np.asarray([[((x[i]-x[j])+1j*(y[i]-y[j]))/(self.rr[i,j]+ \
            np.identity(nat)[i,j]) for j in range(nat)]for i in range(nat)],dtype=np.complex)*(1/np.sqrt(2))
        
            self.x0 = np.asarray([[(z[i]-z[j])/(self.rr[i,j] + \
            np.identity(nat)[i,j]) for j in range(nat)] for i in range(nat)],dtype=float)

            self.index = np.asarray(np.argsort(self.rr)[:,1:nb+1], dtype=np.int)

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


            kv = 1
            
            
            
            D1 = 1*kv*((DDI*1 - 1j*self.rr*kv - (self.rr*kv)**2)/ \
                ((self.rr*kv + np.identity(nat))**3) \
                *np.exp(1j*self.rr*kv)) *(np.ones(nat)-np.identity(nat))   
            D2 = -1*hbar*kv*((DDI*3 - 3*1j*self.rr*kv - (self.rr*kv)**2)/((((kv*self.rr+\
            np.identity(nat))**3))*np.exp(1j*kv*self.rr)))

            # Interaction part
            Di = np.zeros([nat,nat,3,3], dtype = np.complex)

            for i in range(nat):
                for j in range(nat):
                    
                    if (i==j):
                        continue
                    """
                    _________________________________________________________
                    Vacuum interaction (No self-energy)
                    _________________________________________________________
                    """
                        
                    Di[i,j,0,0] = d01m*d1m0*(D1[i,j] -\
                                            self.xp[i,j]*self.xm[i,j]*D2[i,j])
                                            
                    
                    Di[i,j,0,1] = d01m*d00*(-self.xp[i,j]*self.x0[i,j]*D2[i,j]);
                                            
                    Di[i,j,0,2] = d01m*d10*(-self.xp[i,j]*self.xp[i,j]*D2[i,j] );
                                            
                    Di[i,j,1,0] = d00*d1m0*(self.x0[i,j]*self.xm[i,j]*D2[i,j] );
                                            
                    Di[i,j,1,1] = d00*d00*(D1[i,j] + \
                                           self.x0[i,j]*self.x0[i,j]*D2[i,j]);
                                           
                    Di[i,j,1,2] = d00*d10*(self.x0[i,j]*self.xp[i,j]*D2[i,j] );
                                            
                    Di[i,j,2,0] = d01*d1m0*(-self.xm[i,j]*self.xm[i,j]*D2[i,j]);
                                            
                    Di[i,j,2,1] = d01*d00*(-self.xm[i,j]*self.x0[i,j]*D2[i,j]);
                                            
                    Di[i,j,2,2] = d01*d10*(D1[i,j]- \
                                            self.xm[i,j]*self.xp[i,j]*D2[i,j])


            if self.typ == 'L' and not PAIRS:
                try:


                    from sigmaMatrix import returnForLambda
                    self.D = returnForLambda(Di, np.array(self.index, dtype=np.int), nb)
                except:
                    print("sigmaMatrix.pyx is not compiled! Using pure python to form Sigma-matrix (Very Slow).")

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

            elif self.typ == 'L' and PAIRS:


                dim = nat + 2 * nat * (nat - 1)
                self.D = np.zeros([dim, dim], dtype=np.complex)

                possible = False
                for counterOne in range(dim):
                    for counterTwo in range(dim):

                        if counterOne < nat:
                            m1 = 2
                            n1 = counterOne
                            n1a = counterTwo

                        elif counterOne >= nat:
                            n1a = (counterOne - nat) // (2 * nat)
                            n1 = (counterOne - nat ) % (2 * nat) // 2
                            m1 = (counterOne - nat ) % (2 * nat) % 2
                            if n1a>=n1:
                                n1a += 1


                        if counterTwo < nat:
                            m2 = 2
                            n2 = counterTwo
                            n2a = counterOne
                        elif counterTwo >= nat:
                            n2a = (counterTwo - nat ) // (2 * nat)
                            n2 = (counterTwo - nat ) % (2 * nat) // 2
                            m2 = (counterTwo - nat ) % (2 * nat) % 2
                            if n2a>=n2:
                                n2a += 1

                        if n1==n2: continue

                        if (counterOne < nat) != (counterTwo < nat):
                            if counterOne < nat:
                                if counterOne == n2a:
                                    possible = True
                            elif counterTwo < nat:
                                if counterTwo == n1a:
                                    possible = True


                        if (n1 == n2a and n2 == n1a) or possible:
                            possible = False
                            self.D[counterOne, counterTwo] = Di[n1, n2, m1, m2]

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


        def S_matrix_for_Lambda(self):

            """
            Method calculates S matrix for Lambda atom. There are two global options indicated by changing global
            parameter SINGLE_RAMAN to 1 or 0. For 0 it calculates S matrix elements only for Rayleigh
            channel (without change of atomic polarization). For 1 it calculates these ... for changing polarization of
            single atom in a whole atomic chain.

            :return:
            fullTransmittance and fullReflection
            """

            nat = self.nat
            nb = self.nb
            k=1
            kv=1
            c=1
            dim = nat*3**nb
            nof = len(self.freqs)
            
            #Decay rate for Lambda atom with respect of guided modes
            gd = VACUUM_DECAY * np.ones(nat)

            self.gd_full = gd[0]

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


                #ddLeftF_0p[i,j]:
                #i contains the atom which is excited now and state of its neighbours
                #j is the atom which jumps
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
                # Summation over single-jump Raman channels

                Tmatrix = lambda A: np.square(np.absolute(A))
                Tmatrix_reduce = lambda A: np.add.reduce(np.square(np.absolute(A)))
                Tmatrix_reduce = lambda A: np.dot(A, np.conj(A))
                DenMatrix_spur = lambda A, B: np.diag(np.dot(np.transpose(A), np.conj(B)))


            #Loop over atoms, from which the photon emits
            for i in range(nat):

                ddRight[(i+1)*3**nb-1] = d00 * self.e_forward[i]

                """
                ________________________________________________________________
                Rayleigh channels (m = +1 -> m'= +1, p = +1 -> p' = +1)
                ________________________________________________________________
                """
                # Rayleigh
                ddLeftF_pm[(i+1)*3**nb-1] = d00 * self.e_forward[i]
                ddLeftB_pm[(i+1)*3**nb-1] = d00 * self.e_backward[i]
                # Faraday
                ddLeftF_pp[(i+1)*3**nb-1] = d00 * 0
                ddLeftB_pp[(i+1)*3**nb-1] = d00 * 0

                if SINGLE_RAMAN:
                    """
                    ________________________________________________________________
                    Raman channels
                    ________________________________________________________________
                                                Indices:
                            ddLeft<direction>_xy
                            direcrion - F or B - forward or backward, consequently
                            x - final atom state
                            y - final photon state
                    """
                    #loop over atoms, which change their pseudospin projection
                    #they are neighbours of atom, from which final photon leaves
                    counter = 0#I need to count neighbours, not atom1
                    # jp is an atom in cluster and I do loop over this cluster

                    for jp in np.sort(np.append(np.asarray([i],dtype=np.int),self.index[i,:])): #Neighbours + Excited atom

                        #PAY ATTENTION: 'elif' block is on top, 'if' block is on bottom!

                        # if the atom which change its orientation isn't the one, from which photon emits, do the loop
                        # in other words, 'if' block is for amplitudes like <g=-1,0,+1, e|V|g,+1>
                        #if jp!=i:
                        #    jc=counter

                        # if the atom which change its orientation is the one, from which photon emits then do elif
                        # in other words, 'elif' block is for amplitudes like <e|V|g=-1,0,+1>
                        if jp==i:
                            index_0 = (i+1)*3**nb-1 # combination number of neighbours state (all two's)
                            index_m = (i+1)*3**nb-1 #

                            ddLeftF_0m[index_0, jp] = d00 * 0
                            ddLeftB_0m[index_0, jp] = d00 * 0

                            ddLeftF_0p[index_0, jp] = d00 * 0
                            ddLeftB_0p[index_0, jp] = d00 * 0

                            ddLeftF_mm[index_m, jp] = d00 * 0
                            ddLeftB_mm[index_m, jp] = d00 * 0

                            ddLeftF_mp[index_m, jp] = d00 * self.e_forward[i]
                            ddLeftB_mp[index_m, jp] = d00 * self.e_backward[i]

                            continue



                        #here's non-trivial part.
                        #the combination number of (s_1 ... s_p ... s_n) is \sum_{s_p} (2-s_p) 3^(p)
                        #thus, the combination numbers of chain with lacuna in p are:
                        index_0 = (i+1)*3**nb-1 - (2-1)*(3**(counter)) #for <0>-lacuna
                        index_m = (i+1)*3**nb-1 - (2-0)*(3**(counter)) #for <-1>-lacuna

                        ddLeftF_0m[index_0,jp] = +d00 * self.e_forward[i]
                        ddLeftB_0m[index_0,jp] = +d00 * self.e_backward[i]
                    
                        ddLeftF_0p[index_0,jp] = +d00 * 0
                        ddLeftB_0p[index_0,jp] = +d00 * 0
                    
                        ddLeftF_mm[index_m,jp] = +d01 * self.e_forward[i]
                        ddLeftB_mm[index_m,jp] = +d01 * self.e_backward[i]

                        # atom jp in -1 state, atom i in +1 state, photon in -1 state
                        ddLeftF_mp[index_m,jp] = +d00 * 0
                        ddLeftB_mp[index_m,jp] = +d01 * 0
                        counter += 1


            self.fullTransmittance = np.zeros(len(self.freqs), dtype=float)
            self.fullReflection = np.zeros(len(self.freqs), dtype=float)

            from wrap_bypass import get_solution
            #self.D = -self.D
            Resolventa = get_solution(dim, len(self.freqs), nat, self.D, ddRight, self.freqs, gd[0], RABI * np.ones(dim, dtype=np.float32), DC) \
                         * 2 * np.pi * hbar * kv/c

            dot = lambda a,b: np.dot(np.conj(a),b)
            TF_pm = dot(ddLeftF_pm, Resolventa)
            TB_pm = dot(ddLeftB_pm, Resolventa)

            TF_pp = dot(ddLeftF_pp, Resolventa)
            TB_pp = dot(ddLeftB_pp, Resolventa)

            self.Transmittance = np.ones(len(self.freqs), dtype=np.complex64) + 1j * TF_pm
            self.Reflection = 1j * TB_pm

            self.iTransmittance = 1j * TF_pp
            self.iReflection = 1j * TB_pp


            TF2_0m = dot(np.transpose(ddLeftF_0m), Resolventa)
            TB2_0m = dot(np.transpose(ddLeftB_0m), Resolventa)

            TF2_0p = dot(np.transpose(ddLeftF_0p), Resolventa)
            TB2_0p = dot(np.transpose(ddLeftB_0p), Resolventa)

            TF2_mm = dot(np.transpose(ddLeftF_mm), Resolventa)
            TB2_mm = dot(np.transpose(ddLeftB_mm), Resolventa)

            TF2_mp = dot(np.transpose(ddLeftF_mp), Resolventa)
            TB2_mp = dot(np.transpose(ddLeftB_mp), Resolventa)

            dm = np.zeros([4,4,nof], dtype=np.complex64)

            print(self.e_forward, self.e_backward)
            """
            Density Matrix of light 
            0 - forward, minus
            1 - forward, plus
            2 - backward, minus
            3 - backward, plus
            """

            dm[0, 0, :] = DenMatrix_spur(TF2_0m, TF2_0m) + DenMatrix_spur(TF2_mm, TF2_mm) \
                          + abs(np.ones(nof, dtype=np.complex64) - 1j* TF_pm) ** 2
            dm[1, 1, :] = DenMatrix_spur(TF2_0p, TF2_0p) + DenMatrix_spur(TF2_mp, TF2_mp) + abs(TF_pp) ** 2
            dm[2, 2, :] = DenMatrix_spur(TB2_0m, TB2_0m) + DenMatrix_spur(TB2_mm, TB2_mm) + abs(TB_pm) ** 2
            dm[3, 3, :] = DenMatrix_spur(TB2_0p, TB2_0p) + DenMatrix_spur(TB2_mp, TB2_mp) + abs(TB_pp) ** 2

            dm[0, 1, :] = DenMatrix_spur(TF2_0m, TF2_0p) + DenMatrix_spur(TF2_mm, TF2_mp) \
                          + (np.ones(nof, dtype=np.complex64) - 1j* TF_pm) * np.conj(1j*TF_pp)
            dm[0, 2, :] = DenMatrix_spur(TF2_0m, TB2_0m) + DenMatrix_spur(TF2_mm, TB2_mm) \
                          + (np.ones(nof, dtype=np.complex64) - 1j* TF_pm) * np.conj(1j*TB_pm)
            dm[0, 3, :] = DenMatrix_spur(TF2_0m, TB2_0p) + DenMatrix_spur(TF2_mm, TB2_mp) \
                          + (np.ones(nof, dtype=np.complex64) - 1j* TF_pm) * np.conj(1j*TB_pp)
            dm[1, 0, :] = np.conj(dm[0, 1, :])
            dm[2, 0, :] = np.conj(dm[0, 2, :])
            dm[3, 0, :] = np.conj(dm[0, 3, :])

            dm[1, 2, :] = DenMatrix_spur(TF2_0p, TB2_0m) + DenMatrix_spur(TF2_mp, TB2_mm) + TF_pp*np.conj(TB_pm)
            dm[1, 3, :] = DenMatrix_spur(TF2_0p, TB2_0p) + DenMatrix_spur(TF2_mp, TB2_mp) + TF_pp*np.conj(TB_pp)
            dm[2, 1, :] = np.conj(dm[1, 2, :])
            dm[3, 1, :] = np.conj(dm[1, 3, :])

            dm[2, 3, :] = DenMatrix_spur(TB2_0m, TB2_0p) + DenMatrix_spur(TB2_mm, TB2_mp) + TB_pm*np.conj(TB_pp)
            dm[3, 2, :] = np.conj(dm[2, 3, :])

            self.fullTransmittance = dm[0, 0, :] + dm[1, 1, :]

            self.fullReflection = dm[2, 2, :] + dm[3, 3, :]

            self.SideScattering = 1 - self.fullTransmittance - self.fullReflection


            self.dm = dm
            self.TF2_0m = TF2_0m
            self.TB2_0m = TB2_0m

            self.TF2_0p = TF2_0p
            self.TB2_0p = TB2_0p

            self.TF2_mm = TF2_mm
            self.TB2_mm = TB2_mm

            self.TF2_mp = TF2_mp
            self.TB2_mp = TB2_mp
            self.TF_pp = TF_pp
            self.TB_pp = TB_pp


            print('\n')


        def reflection_calc(self):

            if self.typ == 'L' and not PAIRS:
                self.S_matrix_for_Lambda()
            else:
                raise TypeError
                
            
        def visualize(self,addit='',save=True):
            import matplotlib.pyplot as plt
            
            plt.plot(self.freqs, (abs(self.Reflection) ** 2), 'r--', label='Reflection, elastic', lw=1.5)
            plt.plot(self.freqs, abs(self.Transmittance) ** 2, 'm--', label='Trasmission, elastic', lw=1.5)
            plt.plot(self.freqs, self.fullReflection, 'r-', label='Reflection, total', lw=1.5)
            plt.plot(self.freqs, self.fullTransmittance, 'm-', label='Transmission, total', lw=1.5)
            plt.legend()
            plt.xlabel('Detuning, $\gamma$',fontsize=16)
            #plt.ylabel('R,T',fontsize=16)
            plt.savefig('TnR.png',dpi=700)
            plt.show()
            plt.clf()

            #plt.plot(self.deltaP, np.gradient(np.angle(self.Reflection) ), 'r-',label='arg R',lw=1.5)
            #plt.plot(self.deltaP, 2*np.pi*np.gradient(np.angle(self.Transmittance)), 'm-', label='arg T',lw=1.5)
            #plt.legend()
            #plt.xlabel('Detuning, $\gamma$',fontsize=16)
            #plt.ylabel('R,T',fontsize=16)
            #plt.savefig('TnR.png',dpi=700)
            #plt.show()
            #plt.clf()

            #plt.plot(self.deltaP,(abs(self.iReflection)**2 ), 'r-',label='R',lw=1.5)
            #plt.plot(self.deltaP, abs(self.iTransmittance)**2, 'm-', label='T',lw=1.5)
            #plt.legend()
            #plt.xlabel('Detuning, $\gamma$',fontsize=16)
            #plt.ylabel('R,T',fontsize=16)
            #plt.savefig('TnR.png',dpi=700)
            #plt.show()
            #plt.clf()

            if self.typ == 'V':
                plt.plot(self.freqs, (abs(self.iReflection) ** 2), 'r-', label='R', lw=1.5)
                plt.plot(self.freqs, abs(self.iTransmittance) ** 2, 'm-', label='T', lw=1.5)
                plt.legend()
                plt.xlabel('Detuning, $\gamma$',fontsize=16)
                plt.ylabel('R,T',fontsize=16)
                plt.savefig('TnR_rot.png',dpi=700)
                plt.show()
                plt.clf()

            if self.typ == 'L' and SINGLE_RAMAN==True and False:
                plt.plot(self.freqs, self.fullReflection, 'r-', label='R', lw=1.5)
                plt.plot(self.freqs, self.fullTransmittance, 'm-', label='T', lw=1.5)
                plt.legend()
                plt.xlabel('Detuning, $\gamma$',fontsize=16)
                plt.ylabel('R,T',fontsize=16)
                plt.savefig('TnR_full.png',dpi=700)
                plt.show()
                plt.clf()

            
            plt.xlabel('Detuning, $\gamma$')
            plt.ylabel('Loss')
            plt.title('Loss')
            plt.plot(self.freqs, self.SideScattering, 'g-', lw=1.5)
            plt.show()
            plt.clf()
            
 #scale constants

        def simple_visor(self):
            import matplotlib.pyplot as plt

            plt.plot(self.freqs, (abs(self.iReflection) ** 2), 'r-', label='R', lw=1.5)
            plt.plot(self.freqs, abs(self.iTransmittance) ** 2, 'm-', label='T', lw=1.5)

            #plt.plot(self.deltaP, abs(self.fullReflection) , 'r--', label='R', lw=1.5)
            #plt.plot(self.deltaP, abs(self.fullTransmittance) , 'm--', label='T', lw=1.5)
            #plt.plot(self.deltaP, self.SideScattering, 'g-', lw=1.5)

            plt.legend()
            plt.xlabel('Detuning, $\gamma$', fontsize=16)
            plt.ylabel('R,T', fontsize=16)
            plt.savefig('TnR.png', dpi=700)
            plt.show()
            plt.clf()

            plt.plot(self.freqs, self.fullReflection - (abs(self.iReflection) ** 2) - (abs(self.Reflection) ** 2), 'r-', label='R, Raman inelastic', lw=1.5)
            plt.plot(self.freqs, self.fullTransmittance - abs(self.iTransmittance) ** 2 - abs(self.Transmittance) ** 2, 'm-', label='T, Raman inelastic', lw=1.5)

            #plt.plot(self.deltaP, abs(self.fullReflection) , 'r--', label='R', lw=1.5)
            #plt.plot(self.deltaP, abs(self.fullTransmittance) , 'm--', label='T', lw=1.5)
            #plt.plot(self.deltaP, self.SideScattering, 'g-', lw=1.5)

            plt.legend()
            plt.xlabel('Detuning, $\gamma$', fontsize=16)
            plt.ylabel('R,T', fontsize=16)
            plt.savefig('TnR.png', dpi=700)
            plt.show()
            plt.clf()

        def dm_visualize(self, om = -1):
            if om == -1:
                om = len(self.freqs) / 2
            import matplotlib.pyplot as plt
            plt.imshow(np.real(self.dm[:,:,om]))
            plt.show()

            plt.imshow(np.imag(self.dm[:,:,om]))
            plt.show()



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

freq = np.linspace(-7,7, 1980)*gd

#Validation (all = 1 iff ful theory, except SINGLE_RAMAN)
WAIST = 16.
OMEGA  = 1.
DDI = 1
VACUUM_DECAY = 1# = 0 iff assuming only decay into fundamental mode, =1 iff decay into fundamental and into radiation
SINGLE_RAMAN = True
OPPOSITE_SCATTERING = False
RAMAN_BACKSCATTERING = False
PAIRS = False
FIX_RANDOM = True
RABI = 2#4#4+1e-16 #Rabi frequency
DC = -4 #Rabi detuning
SHIFT = 0
RABI_HYP = False
SEED = 12


if not SINGLE_RAMAN and RAMAN_BACKSCATTERING:
    SINGLE_RAMAN = True

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
            'nat':16, #number of atoms
            'nb':0, #number of neighbours in raman chanel (for L-atom only)
            's':'square', #Stands for atom positioning : chain, nocorrchain and doublechain
            'dist':0.,  # sigma for displacement (choose 'chain' for gauss displacement., \lambda/2 units)
            'd' : 0, # distance from center
            'l0': 2.00/2, # mean distance between atoms (in lambda_m /2 units)
            'freqs':freq,  # array of freq
            'typ':'L',  # L or V for Lambda and V atom resp.
            'ff': 0.3 #filling factor (for ff_chain only)
            }

    dtun = 10
    import matplotlib.pyplot as plt

    PAIRS = False
    WAIST = 4.
    SE8 = ensemble(**args)
    SE8.generate_ensemble()

    WAIST = 8.
    SE16 = ensemble(**args)
    SE16.generate_ensemble()

    #plt.plot(freq, abs(SE8.Transmittance - SE8.Reflection) **2 )
    #plt.plot(freq, 1 - abs(SE8.Transmittance - SE8.Reflection) ** 2)
    #plt.plot(freq, abs(SE16.Transmittance - SE16.Reflection) **2 )
    #plt.plot(freq, 1 - abs(SE16.Transmittance - SE16.Reflection) ** 2)

    #plt.show()

    plt.plot(freq, abs(SE8.Transmittance-SE8.Reflection)**2)
    plt.plot(freq, abs(SE8.Reflection+SE8.Transmittance) ** 2)
    plt.plot(freq, 1-abs(SE8.Transmittance-SE8.Reflection)**2-abs(SE8.Reflection+SE8.Transmittance) ** 2)

    plt.plot(freq, abs(SE16.Transmittance)**2)
    plt.plot(freq, abs(SE16.Reflection) ** 2)
    plt.plot(freq, 1-abs(SE16.Transmittance)**2-abs(SE16.Reflection)**2)

    plt.show()


    """
    nrandoms = 100
    nof = len(freq)
    ref_raw = np.empty([nof, nrandoms], dtype = np.float32)
    tran_raw = np.empty([nof, nrandoms], dtype=np.float32)

    RADIATION_MODES_MODEL = 1
    for i in range(nrandoms):
        SEED = i
        SE0 = ensemble(**args)
        SE0.L = 2 * np.pi
        SE0.generate_ensemble()
        print('With DDI, Epoch %d / ' % (i + 1), nrandoms)

        ref_raw[:, i] = abs(SE0.Reflection)**2
        tran_raw[:, i] = abs(SE0.Transmittance)**2

    ref_av = np.average(ref_raw, axis=1)
    tran_av = np.average(tran_raw, axis=1)

    RADIATION_MODES_MODEL = 0
    for i in range(nrandoms):
        SEED = i
        SE0 = ensemble(**args)
        SE0.L = 2 * np.pi
        SE0.generate_ensemble()
        print('Without DDI, Epoch %d / ' % (i + 1), nrandoms)

        ref_raw[:, i] = abs(SE0.Reflection)**2
        tran_raw[:, i] = abs(SE0.Transmittance)**2

    ref_av_nodd = np.average(ref_raw, axis=1)
    tran_av_nodd = np.average(tran_raw, axis=1)

    args['s'] = 'chain'
    args['dist'] = 0.5

    RADIATION_MODES_MODEL = 1
    for i in range(nrandoms):
        SEED = i
        SE0 = ensemble(**args)
        SE0.L = 2 * np.pi
        SE0.generate_ensemble()
        print('Gauss With DDI, Epoch %d / ' % (i + 1), nrandoms)

        ref_raw[:, i] = abs(SE0.Reflection)**2
        tran_raw[:, i] = abs(SE0.Transmittance)**2

    ref_ga = np.average(ref_raw, axis=1)
    tran_ga = np.average(tran_raw, axis=1)

    RADIATION_MODES_MODEL = 0
    for i in range(nrandoms):
        SEED = i
        SE0 = ensemble(**args)
        SE0.L = 2 * np.pi
        SE0.generate_ensemble()
        print('Gauss Without DDI, Epoch %d / ' % (i + 1), nrandoms)

        ref_raw[:, i] = abs(SE0.Reflection)**2
        tran_raw[:, i] = abs(SE0.Transmittance)**2

    ref_ga_nodd = np.average(ref_raw, axis=1)
    tran_ga_nodd = np.average(tran_raw, axis=1)

    plt.title('With DDI')

    plt.plot(freq, ref_av)
    plt.plot(freq, tran_av)

    plt.legend()
    plt.show()

    plt.title('Without DDI')

    plt.plot(freq, ref_av_nodd)
    plt.plot(freq, tran_av_nodd)

    plt.legend()
    plt.show()

    plt.title('Gauss with DDI')

    plt.plot(freq, ref_ga)
    plt.plot(freq, tran_ga)

    plt.legend()
    plt.show()

    plt.title('Gauss without DDI')

    plt.plot(freq, ref_ga_nodd)
    plt.plot(freq, tran_ga_nodd)

    plt.legend()
    plt.show()
    """
    #RABI = dtun+4
    #DC = -dtun+4

    #SE1 = ensemble(**args)
    #SE1.L = 2*np.pi
    #SE1.generate_ensemble()

    #PAIRS = False
    #RABI = 0
    #DC = 0
    #SE0f = ensemble(**args)
    #SE0f.L = 2*np.pi
    #SE0f.generate_ensemble()

    #RABI = dtun+4
    #DC = -dtun+4

    #SE1f = ensemble(**args)
    #SE1f.L = 2*np.pi
    #SE1f.generate_ensemble()
    #Dists = [0., .01, .015, .02, .1, -0.01, -.015, -.02, -.1]
    #for dist in Dists:
    #   args['l0'] = 1. + dist
    #   #args['dist'] = 10*dist
    #   SE0 = ensemble(**args)
    #   SE0.L = 2*np.pi
    #   SE0.generate_ensemble()
    #   if dist>0:
    #        line = 'r-'
    # #   else:
    # #       line = 'b-'
    # #   #plt.plot(freq, SE0.fullTransmittance, 'b')
    # #   plt.plot(freq, abs(SE0.Transmittance)**2, label=str(10*dist))
    # #   plt.plot(freq, abs(SE0.Reflection)**2, 'r-')
        #plt.plot(freq, SE0.fullReflection, 'r-')
        #plt.plot(freq, SE0.SideScattering, 'g')
        #plt.show()

    #plt.plot(freq, SE0f.fullTransmittance, 'b--')
    #plt.plot(freq, SE0f.fullReflection, 'r--')
    #plt.plot(freq, SE0f.SideScattering, 'g--')
    #plt.legend()
    #plt.show()


    #plt.plot(freq, SE1.fullTransmittance, 'b')
    #plt.plot(freq, SE1.fullReflection, 'r')
    #plt.plot(freq, SE1.SideScattering, 'g')

    #plt.plot(freq, SE1f.fullTransmittance, 'b-')
    #plt.plot(freq, SE1f.fullReflection, 'r-')
    #plt.plot(freq, SE1f.SideScattering, 'g-')

