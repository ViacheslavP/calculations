def get_solution_pairs(dim, nof, noa, Sigma, ddRight, freq, gamma, rabi, dc, edge=False):
    import numpy as np
    from scipy.sparse import lil_matrix, csr_matrix
    from scipy.sparse.linalg import lgmres as spsolve
    import sys
    scV = np.empty((dim, nof), dtype=np.complex)
    freq_scaled = (-freq + rabi[0] ** 2 / (4 * (freq - dc)) - 0.5j * gamma)
    oned = lil_matrix((dim, dim), dtype=np.complex)

    for i in range(dim):
        oned[i,i] = 1.

    Sigma = csr_matrix(Sigma, dtype=np.complex)
    oned = csr_matrix(oned, dtype=np.complex)


    for i,om in enumerate(freq_scaled):
        resolvent = (om-1.)*oned + Sigma
        scV[:,i], exitCode = spsolve(resolvent, ddRight)
        try:
            assert exitCode == 0
        except AssertionError:
            if exitCode > 0:
                print(f'Convergence not achieved. Step {i}, Number of iterations {exitCode} \n Continue ...')
            elif exitCode < 0:
                print('Something bad happened. Exitting...')
                assert exitCode == 0
        ist = 100 * (i+1) / nof
        sys.stdout.write("\r%d%%" % ist)
        sys.stdout.flush()
        sys.stdout.write("\033[K")

    return scV

def get_solution(dim, nof, noa, Sigma, ddRight, freq, gamma, rabi, dc):

    import numpy as np


    try:
        if dim <= 1550:
            print("Dimensionality is too small to use CUDA")
            raise ValueError
        import ctypes
        calcalib = ctypes.cdll.LoadLibrary('/home/viacheslav/Projects/PycharmProjects/calculations/core/libbypass.so')
        print("Using CUDA sparsity backend...")
    except:
        scV = np.empty((dim, nof), dtype=np.complex)
        try:
            #TODO: Make this part work!
            raise MemoryError
            resolvent = np.empty((nof, dim,dim), dtype=np.complex)
            freq_scaled = (-freq + rabi[0] ** 2 / (4 * (freq - dc)) - 0.5j * gamma)

            resolvent = np.broadcast_to(freq_scaled, (nof, dim, dim)) + np.broadcast_to(Sigma, (nof, dim, dim))
            print(np.shape(resolvent))
            scV = np.transpose(np.linalg.solve(resolvent, ddRight))

        except MemoryError:
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import spsolve

            print("Memory Error was found. Using iterations:")
            for i, om in enumerate(freq):
                resolvent = csr_matrix(np.eye(dim) * (-om + rabi[0] ** 2 / (4 * (om - dc)) - 0.5j * gamma) + Sigma)
                scV[:, i] = spsolve(resolvent, ddRight)

        return scV

    else:
        scV = np.empty(dim * nof, dtype=np.complex64)

        calcalib.scProblem_solve.argtypes = [ctypes.c_int,
                                         ctypes.c_int,
                                         ctypes.c_int,
                                         np.ctypeslib.ndpointer(np.complex64, ndim=1, flags='C'),
                                         np.ctypeslib.ndpointer(np.complex64, ndim=1, flags='C'),
                                         np.ctypeslib.ndpointer(np.float32, ndim=1, flags='C'),
                                         ctypes.c_float,
                                         np.ctypeslib.ndpointer(np.float32, ndim=1, flags='C'),
                                         ctypes.c_float,
                                         np.ctypeslib.ndpointer(np.complex64, ndim=1, flags='C')]


        calcalib.scProblem_solve(dim,
                             nof,
                             noa,
                             np.reshape(np.eye(dim)+Sigma, (dim*dim)).astype(np.complex64),
                             ddRight.astype(np.complex64),
                             freq.astype(np.float32),
                             gamma,
                             rabi.astype(np.float32),
                             dc,
                             scV)

        return np.transpose(np.reshape(scV, (nof, dim)))


