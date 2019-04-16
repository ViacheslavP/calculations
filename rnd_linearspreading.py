import sys
sys.path.insert(0, 'core/')
import one_D_scattering as ods
from wave_pack import convolution, delay
from wave_pack import gauss_pulse as pulse
import numpy as np
from matplotlib import pyplot as plt

import ctypes
mkl_rt = ctypes.CDLL('libmkl_rt.so')
mkl_get_max_threads = mkl_rt.mkl_get_max_threads
def mkl_set_num_threads(cores):
    mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(cores)))

mkl_set_num_threads(4)
print(mkl_get_max_threads()) # says 4


ods.PAIRS = False
ods.RABI = 0*14
#ods.SINGLE_RAMAN = False
ods.DC = 6#7
SHIFT = +1.81#-0.8#3.56#4.65#4.65#2.80#ods.DC-10.37
freq = np.linspace(-13.5,13.5, 300)
#Pulse duration
pdTime = 2*np.pi
vis = True
noa = 200
sq_reduce = lambda A: np.add.reduce(np.square(np.absolute(A)), axis=1)
ods.RABI_HYP = False


args = {

    'nat': 100,  # number of atoms
    'nb': 0,  # number of neighbours in raman chanel (for L-atom only)
    's': 'chain',  # Stands for atom positioning : chain, nocorrchain and doublechain
    'dist': 0.,  # sigma for displacement (choose 'chain' for gauss displacement.)
    'd': 1.5,  # distance from fiber
    'l0': 2. / 2,  # mean distance between atoms (in lambda_m /2 units)
    'deltaP': freq,  # array of freq
    'typ': 'L',  # L or V for Lambda and V atom resp.
    'ff': 0.3  # filling factor (for ff_chain only)
}


Distances = np.linspace(0.0, 5.5, 200)
Sigma_init = 0.2
ref_arg1 = []
ref_arg2 = []
ref_max = []
ref_zero = []
nrandoms = 30
nof = len(freq)
ref_raw = np.empty([nof, nrandoms], dtype=np.float32)
tran_raw = np.empty([nof, nrandoms], dtype=np.float32)
for dist in Distances:
    for i in range(nrandoms):
        args['dist'] = Sigma_init
        args['l0'] = dist
        SEED = i
        SE0 = ods.ensemble(**args)
        SE0.L = 2 * np.pi
        SE0.generate_ensemble()

        ref_raw[:, i] = abs(SE0.Reflection) ** 2
        tran_raw[:, i] = abs(SE0.Transmittance) ** 2
        print(str(10 * dist) + ' , ' + str(i + 1) + ' / ' + str(nrandoms))
    ref_av = np.average(ref_raw, axis=1)
    tran_av = np.average(tran_raw, axis=1)

    ref_arg1.append(freq[np.argsort(ref_av)[nof - 1]])
    ref_arg2.append(freq[np.argsort(ref_av)[nof - 2]])
    ref_max.append(np.amax(ref_av))
    zero_freq_num = np.argsort(abs(freq))[0]
    ref_zero.append(ref_av[zero_freq_num])

plt.plot(Distances, ref_max, 'b-', label='max')
plt.plot(Distances, ref_zero, 'r-', label='zero')
plt.plot(Distances, ref_arg1, 'ro')
plt.plot(Distances, ref_arg2, 'bo')
plt.legend()
plt.show()