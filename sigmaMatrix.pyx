import numpy as np
cimport numpy as np

DTYPE = np.complex
ctypedef np.complex DTYPE_t

DINT = np.int
ctypedef np.int_t DINT_t

"""
_______________________________________________________________________________
------------------------Sigma Matrix for V ------------------------------------
_______________________________________________________________________________
"""

cdef np.ndarray[DTYPE_t, ndim=2] creturnForV(np.ndarray[DTYPE_t, ndim=4] Di, int nat):
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
---------------------Sigma Matrix for Lambda ----------------------------------
_______________________________________________________________________________
"""

cdef np.ndarray[DINT_t, ndim=2] creturnStates(nb):
    cdef int i,j
    cdef result = np.zeros([3**nb, nb], dtype = DINT)
    for i in range(3**nb):
        for j in range(nb):
            result[i,j] = i // 3**j % 3
    return result


cdef np.ndarray[DTYPE_t, ndim=2] creturnForLambda(np.ndarray[DTYPE_t, ndim=4] Di, int nat, int nb,  np.ndarray[DINT_t, ndim=2] index):
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

def returnForLambda(np.ndarray[DTYPE_t, ndim=4] Di, np.ndarray[DINT_t, ndim=2] index, int nb):
    cdef int n = Di.shape[0]
    return creturnForLambda(Di, n, nb, index)


