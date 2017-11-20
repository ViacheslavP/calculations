import sys
sys.path.insert(0, 'core/')
import one_D_scattering as ods
import numpy as np

import ctypes
mkl_rt = ctypes.CDLL('libmkl_rt.so')
mkl_get_max_threads = mkl_rt.mkl_get_max_threads
def mkl_set_num_threads(cores):
    mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(cores)))

mkl_set_num_threads(4)
print(mkl_get_max_threads()) # says 4

"""
Next 3 functions are: Fourier Integral Transform (FIT), reverse FIT, 
and convolution transformation in this particular task
"""

def FIT(array, freq):
    domega = freq[1]-freq[0]
    FIT_array = np.fft.fft(array)
    time = np.fft.fftfreq(array.size)*2*np.pi/domega
    FIT_array *= domega * np.exp(-1j * freq[0] * time) / (2 * np.pi)
    return time, FIT_array

def revFIT(FIT_array, time):
    dt = time[1]-time[0]
    array = np.fft.rfft(FIT_array)
    freq = np.fft.rfftfreq(FIT_array.size)*2*np.pi/dt
    array *= dt * np.exp(-complex(0, 1) * freq * time[0])
    return freq, array

def convolution(freq, kernel, pulse, vg=0.7):
    im  = np.multiply(kernel, pulse) / vg
    time, fim = FIT(im,freq)
    _sw = np.argsort(time)
    time = time[_sw]
    fim = fim[_sw]
    return time, fim

def gauss_pulse(om, T0):
    """
    Inc. pulse as function of $\omega$ (conj. to time, frequency) (Fourier image)
    T0 - pulse parameter (mean square duration)
    """
    return  np.exp((1j * om * T0 * (1j * om * T0 + 2 * np.pi**2)) / (4*np.pi**2)) * ((2 / np.pi) * T0**2)**(1/4)

def extract_delay(t, f):
    """
    Extracting pulse delay
    """
    return sum(t*abs(f)**2)/sum(abs(f)**2) - pdTime/2

def extract_amplitude(t, f):
    return max(abs(f) ** 2)


if __name__ == '__main__':

    """
    Pulse parameters:
    """
    #Rabi frequency
    ods.RABI = 4.
    ods.SINGLE_RAMAN = False

    #Pulse duration
    pdTime = 2 * np.pi

    #Frequency interval, vast enough
    freq = ods.freq

    #Number of atoms array to iterate
    lenNumbers = 1
    maxAtoms = 100
    numberAtoms = np.linspace(100, maxAtoms+1, lenNumbers, dtype=int)

    #Distance between atoms (regular)
    lenDist = 10
    distbAtoms = np.linspace(0.5, 1.5, lenDist)

    #Arguments of one_D_scattering module
    args = {

    'nat': 20,  # number of atoms
    'nb': 2,  # number of neighbours in raman chanel (for L-atom only)
    's': 'chain',  # Stands for atom positioning : chain, nocorrchain and doublechain
    'dist': 0.,  # sigma for displacement (choose 'chain' for gauss displacement.)
    'd': 1.5,  # distance from fiber
    'l0': 1.000,  # mean distance between atoms (in lambda_m /2 units)
    'deltaP': freq,  # array of freq.
    'typ': 'L',  # L or V for Lambda and V atom resp.
    'ff': 0.3
    }

    delayArray = np.empty([lenNumbers, lenDist],dtype=float)
    amplitudeArray = np.empty([lenNumbers, lenDist], dtype=float)
    approxDelayArray = np.empty([lenNumbers, lenDist],dtype=float)
    for iAtom in range(lenNumbers):
        args['nat'] = numberAtoms[iAtom]
        for jDist in range(lenDist):
            args['l0'] = distbAtoms[jDist]
            print('NoA =', numberAtoms[iAtom], 'Distance, in mode wavelength units = ', 0.5*distbAtoms[jDist])
            elasticSc = ods.ensemble(**args)
            elasticSc.generate_ensemble()
            _t, _f = convolution(freq, elasticSc.Transmittance, gauss_pulse(freq, pdTime), vg=elasticSc.vg)
            delayArray[iAtom, jDist] = extract_delay(_t, _f)
            amplitudeArray[iAtom, jDist] = extract_amplitude(_t, _f)
            approxDelayArray[iAtom, jDist] = np.gradient(np.angle(elasticSc.Transmittance), freq[1]-freq[0])[139]



    np.savez('data/delayFile.npz', amplitude=amplitudeArray, delay=delayArray, numbers = numberAtoms, \
             distances = distbAtoms, approxdelay = approxDelayArray)