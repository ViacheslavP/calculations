import numpy as np
cimport numpy as np

DTYPE = np.complex128
ctypedef np.complex128_t DTYPE_t

DINT = np.int
ctypedef np.int_t DINT_t

DREAL = np.float64
ctypedef np.float64_t DREAL_t


"""
_______________________________________________________________________________
--------------------------Making base simgma-matrix----------------------------
_______________________________________________________________________________
"""

cdef np.ndarray[DTYPE_t, ndim=4] creturnDi(np.ndarray[DREAL_t, ndim=1] x_cords,
                                           np.ndarray[DREAL_t, ndim=1] y_cords,
                                           np.ndarray[DREAL_t, ndim=1] z_cords,
                                           DREAL_t a,
                                           DREAL_t n,
                                           int typ,
                                           int nat,
                                           int RADIATION = 1,
                                           int PARAXIAL = 1):

    cdef np.ndarray[DTYPE_t, ndim=4] Di = np.zeros([nat,nat,3,3], dtype = DTYPE)
    cdef int i,j
    cdef DREAL_t rr,x0,xi,xj,yi,yj,zi,zj


    """
    Scaling Units
    """
    
    cdef DREAL_t gd = 1.
    cdef DREAL_t lambd = 1.
    cdef DREAL_t hbar = 1.
    cdef DREAL_t c = 1.
    cdef DREAL_t d02 = hbar*gd/4*lambd**3

    cdef DTYPE_t xp,xm,D1,D2,Dz
    from bmode import exact_mode
    m = exact_mode(1, n, a)
    cdef DREAL_t kv = 1.
    cdef DREAL_t beta = m.k
    cdef DREAL_t vg = m.vg
    cdef np.ndarray[DREAL_t, ndim=1] r  = np.sqrt(x_cords**2+y_cords**2)
    cdef np.ndarray[DTYPE_t, ndim=1] em = -m.E(r)
    cdef np.ndarray[DTYPE_t, ndim=1] ep = PARAXIAL*-m.dE(r)*(x_cords-1j*y_cords)**2/r/r
    cdef np.ndarray[DTYPE_t, ndim=1] ez = PARAXIAL*m.Ez(r)*(x_cords-1j*y_cords)/r
    cdef np.ndarray[DTYPE_t, ndim=1] emfi,emfjc,epfi,epfjc,embi,embjc,epbi,epbjc
    cdef np.ndarray[DTYPE_t, ndim=2] forward,backward

    for i in range(nat):
        for j in range(nat):

            if (i == j) and (typ == 1):
                continue

            xi = x_cords[i]; xj = x_cords[j];
            yi = y_cords[i]; yj = y_cords[j];
            zi = z_cords[i]; zj = z_cords[j];

            rr = np.sqrt((xi - xj)**2 + (yi - yj)**2 + (zi - zj)**2)
            x0 =  (zi - zj)/rr
            xp = -((xi - xj) + 1j*(yi-yj))/rr
            xm =  ((xi - xj) - 1j*(yi-yj))/rr

            Dz =   \
                    -1*2j*np.pi*kv*np.exp(1j*beta*(abs(x0))*rr)*(1/vg) + \
                    1*2j*np.pi*kv*np.exp(1j*kv*(abs(x0))*rr)*(RADIATION / c)


            if (i != j):
                D1 = RADIATION*kv*((1 - 1j*rr*kv - (rr*kv)**2)/((rr*kv)**3) *np.exp(1j*rr*kv))
                D2 = RADIATION*-1*hbar*kv*((3 - 3*1j*rr*kv - (rr*kv)**2)/((((kv*rr)**3))*np.exp(1j*kv*rr)))
            else:
                D1 = 0.j
                D2 = 0.j

            """
                    ________________________________________________________
                    Guided modes interaction (and self-energy for V atom)
                    ________________________________________________________

                    I am using next notations:
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

            emfi  =  np.conjugate(  np.array([ep[i], ez[i], em[i]], dtype=DTYPE))
            emfjc =                 np.array([ep[j], ez[j], em[j]], dtype=DTYPE)

            epfi  =                 np.array([em[i], ez[i], ep[i]], dtype=DTYPE)
            epfjc =  np.conjugate(  np.array([em[j], ez[j], ep[j]], dtype=DTYPE) )



            embi  =  np.conjugate(  np.array([-ep[i], ez[i], -em[i]], dtype=DTYPE))
            embjc =                 np.array([-ep[j], ez[j], -em[j]], dtype=DTYPE)

            epbi  =                 np.array([-em[i], ez[i], -ep[i]], dtype=DTYPE)
            epbjc =  np.conjugate(  np.array([-em[j], ez[j], -ep[j]], dtype=DTYPE) )

            forward = np.outer(emfi, emfjc) + np.outer(epfi, epfjc)
            backward = np.outer(embi, embjc) + np.outer(epbi, epbjc)


            if abs(zi - zj) < 1e-6:
                Di[i,j,:,:] = 0.5 * (forward + backward) * Dz  #Principal value of integaral: it's corresponds with Fermi's golden rule!

            elif zi>zj:
                Di[i,j,:,:] = forward * Dz

            elif zi<zj:
                Di[i,j,:,:] = backward * Dz

            """
                _________________________________________________________
                     Vacuum interaction (No self-energy)
                _________________________________________________________
            """

            Di[i,j,0,0] = Di[i,j,0,0] + (D1 - xp*xm*D2)
            Di[i,j,0,1] = Di[i,j,0,1] + (-xp*x0*D2);
            Di[i,j,0,2] = Di[i,j,0,2] + (-xp*xp*D2);
            Di[i,j,1,0] = Di[i,j,1,0] + (x0*xm*D2);
            Di[i,j,1,1] = Di[i,j,1,1] + (D1 + x0*x0*D2);
            Di[i,j,1,2] = Di[i,j,1,2] + (x0*xp*D2);
            Di[i,j,2,0] = Di[i,j,2,0] + (-xm*xm*D2);
            Di[i,j,2,1] = Di[i,j,2,1] + (-xm*x0*D2);
            Di[i,j,2,2] = Di[i,j,2,2] + (D1 - xm*xp*D2)

    return d02*Di







"""
_______________________________________________________________________________
-------------------Reshaping base matrix into super-matrix----------------------
_______________________________________________________________________________

"""

"""
_______________________________________________________________________________
--------------------------------for V ----------------------------------------
_______________________________________________________________________________
"""

cdef np.ndarray[DTYPE_t, ndim=2] creturnForV(np.ndarray[DTYPE_t, ndim=4] Di,
                                             int nat):
    cdef np.ndarray[DTYPE_t, ndim=2] D = np.zeros([3*nat,3*nat],dtype=DTYPE)
    cdef int n1,n2,i,j
    for n1 in range(nat): #initial excited
        for n2 in range(nat):#final excited
                for i in range(3):
                     for j in range(3):
                        D[3*n1+j,3*n2+i] = 3*Di[n1,n2,j,i]
    return D

def returnForV(np.ndarray[DTYPE_t, ndim=4] Di):
    cdef int n = Di.shape[0]
    return creturnForV(Di, n)


"""
_______________________________________________________________________________
-------------------------------for Lambda -------------------------------------
_______________________________________________________________________________
"""

cdef np.ndarray[DINT_t, ndim=2] creturnStates(nb):
    cdef int i,j
    cdef result = np.zeros([3**nb, nb], dtype = DINT)
    for i in range(3**nb):
        for j in range(nb):
            result[i,j] = i // 3**j % 3
    return result


cdef np.ndarray[DTYPE_t, ndim=2] creturnForLambda(np.ndarray[DTYPE_t, ndim=4] Di,
                                                  int nat,
                                                  int nb,
                                                  np.ndarray[DINT_t, ndim=2] index):

    cdef np.ndarray[DTYPE_t, ndim=2] D = np.zeros([3**nb*nat,3**nb*nat],dtype=DTYPE)
    cdef np.ndarray[DINT_t, ndim=3] st = 2*np.ones([nat, nat, 3**nb],dtype=DINT)


    cdef np.ndarray[DINT_t, ndim=2] state = creturnStates(nb)


    cdef int n1,n2,i,j,k,ki1
    for n1 in range(nat):
        k=0
        for n2 in np.sort(index[n1,:]):
            for i in range(3**nb):
                st[n1,n2,i] = state[i,k]
            k+=1

    cdef int clock = 1
    for n1 in range(nat): #initial excited
        for n2 in range(nat):#final excited
                for i in range(3**nb):
                     for j in range(3**nb):
                        clock = 1
                        for ki1 in np.append(index[n1,:],index[n2,:]):

                            if ki1 == n1 or ki1 == n2: continue;
                            if (st[n1,ki1,i] != st[n2,ki1,j]):
                                clock = 0
                                break
                        if clock == 1:
                            D[n1*3**nb+i,n2*3**nb+j] =  \
                                Di[n1,n2,st[n1,n2,i],st[n2,n1,j]]
    return D

def returnForLambda(np.ndarray[DTYPE_t, ndim=4] Di,
                    np.ndarray[DINT_t, ndim=2] index,
                    int nb):
    cdef int nn = Di.shape[0]
    return creturnForLambda(Di, nn, nb, index)


"""
_______________________________________________________________________________
--------------------------Calculation of amplitudes----------------------------
_______________________________________________________________________________
"""

cdef np.ndarray[DTYPE_t, ndim=2] camplitudesForV(np.ndarray[DTYPE_t, ndim=2] D,
                                                 np.ndarray[DTYPE_t, ndim=2] zij,
                                                 DREAL_t _from,
                                                 DREAL_t _to,
                                                 DINT_t _num,
                                                 DINT_t nat):
    import sys

    """
            ________________________________________________________________
            Definition of channels of Scattering
            ________________________________________________________________
    """
    cdef np.ndarray[DTYPE_t, ndim=1] ddLeftF = np.zeros(3*nat, dtype=np.complex);
    cdef np.ndarray[DTYPE_t, ndim=1] ddLeftB = np.zeros(3*nat, dtype=np.complex);
    cdef np.ndarray[DTYPE_t, ndim=1] ddLeftFm = np.zeros(3*nat, dtype=np.complex);
    cdef np.ndarray[DTYPE_t, ndim=1] ddLeftBm = np.zeros(nat*3, dtype=np.complex);
    cdef np.ndarray[DTYPE_t, ndim=1] ddRight = np.zeros(3*nat, dtype=np.complex);
    cdef np.ndarray[DTYPE_t, ndim=2] One = (np.identity(3*nat)) #Unit in V-atom
    cdef np.ndarray[DTYPE_t, ndim=2] Sigmad = (np.identity(nat*3,dtype=np.complex))
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

        omega = deltaP[k]

        Sigma = Sigmad+omega*One
        V =  1*(- self.D/hbar/lambd/lambd )
        ResInv = Sigma + V
        Resolventa =  np.linalg.solve(ResInv,ddRight)

        self.Transmittance[k] = 1.+(-1j/hbar/(self.vg))*np.dot(Resolventa,ddLeftF)*2*np.pi*hbar*kv
        self.Reflection[k] = (-1j/hbar/(self.vg)))*np.dot(Resolventa,ddLeftB)*2*np.pi*hbar*kv


        self.iTransmittance[k] = (-1j/hbar/(self.vg))*np.dot(Resolventa,ddLeftFm)*2*np.pi*hbar*kv
        self.iReflection[k] =  (-1j/hbar/(self.vg)))*np.dot(Resolventa,ddLeftBm)*2*np.pi*hbar*kv

        self.SideScattering[k] =1 - abs(self.Transmittance[k])**2-abs(self.Reflection[k])**2-\
                                          abs(self.iTransmittance[k])**2-abs(self.iReflection[k])**2

        ist = 100*k / len(self.deltaP)
        sys.stdout.write("\r%d%%" % ist)
        sys.stdout.flush()



    print('\n')

