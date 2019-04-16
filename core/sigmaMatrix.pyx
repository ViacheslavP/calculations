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
---------------for Lambda (with pairs and without neighbours)------------------
_______________________________________________________________________________
"""

cdef np.ndarray[DTYPE_t, ndim=2] creturnPairsForLambda(np.ndarray[DTYPE_t, ndim=4] Di,
                                                  int nat):

    cdef int dim  = nat + 2 * nat * (nat - 1)
    cdef np.ndarray[DTYPE_t, ndim=2] D = np.zeros([dim,dim],dtype=DTYPE)
    cdef int counterOne, counterTwo,n1,n2,n1a,n2a,m1,m2
    for counterOne in range(dim):
        for counterTwo in range(dim):

            if counterOne < nat:
                m1 = 2
                n1 = counterOne
                n1a = counterTwo
            elif counterOne >= nat:
                n1a = (counterOne - nat) // (2*nat)
                n1 =  (counterOne - nat) % (2*nat) // 2
                m1 =  (counterOne - nat) % (2*nat) % 2


            if counterTwo < nat:
                m2 = 2
                n2 = counterTwo
                n2a = counterOne
            elif counterTwo >= nat:
                n2a = (counterTwo - nat) // (2*nat)
                n2 =  (counterTwo - nat) % (2*nat) // 2
                m2 =  (counterTwo - nat) % (2*nat) % 2

            if n1 == n2a and n2 == n1a:
                D[counterOne, counterTwo] = Di[n1,n2,m1,m2]



def returnPairsForLambda(np.ndarray[DTYPE_t, ndim=4] Di):
    cdef int nn = Di.shape[0]
    return creturnPairsForLambda(Di, nn)




