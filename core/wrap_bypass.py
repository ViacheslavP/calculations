def get_solution(dim, nof, noa, Sigma, ddRight, freq, gamma, rabi, dc):

    import ctypes
    import numpy as np

    calcalib = ctypes.cdll.LoadLibrary('/home/viacheslav/Projects/PycharmProjects/calculations/core/libbypass.so')
    scV = np.empty(dim*nof, dtype=np.complex64)

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