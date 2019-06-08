import numpy as np
from scipy.integrate import simps
"""
Next 3 functions are: Fourier Integral Transform (FIT), reverse FIT, 
and convolution transformation in this particular task
"""

def FIT(array, freq):
    domega = freq[1]-freq[0]
    FIT_array = np.fft.fft(array)
    time = np.fft.fftfreq(array.size)*2*np.pi/domega
    FIT_array *= domega * np.exp(-1j * (freq[0]+freq[-1])/2 * time) / (2 * np.pi)
    return time, FIT_array

def revFIT(FIT_array, time):
    dt = time[1]-time[0]
    array = np.fft.rfft(FIT_array)
    freq = np.fft.rfftfreq(FIT_array.size)*2*np.pi/dt
    array *= dt * np.exp(-complex(0, 1) * freq * time[0])
    return freq, array

def convolution(freq_c, kernel, pulse, vg=0.7):
    im  = np.multiply(kernel, pulse)
    time, fim = FIT(im,freq_c)
    _sw = np.argsort(time)
    time = time[_sw]
    fim = fim[_sw]
    return time, fim


def rect_pulse(om, gamma):
    T0 = 1/gamma
    return 1 / (1j * om) * (np.exp(1j*om*T0) - 1)

def gauss_pulse(om, gamma):
    """
    Inc. pulse as function of $\omega$ (conj. to time, frequency) (Fourier image)
    T0 - pulse parameter (mean square duration)
    """
    T0 = 1/gamma
    return  np.exp((1j * om * T0 * (1j * om * T0 + 2 * np.pi**2))/ (4*np.pi**2)) * ((2 / np.pi) * T0**2)**(1/4)

def inverse_pulse(om, gamma):
    return 1/(om - 1j*gamma/2)

def gauss_pulse_v2(om, tau):
    #fourier of Exp[-2Ln2*t^2/tau^2]
    return np.exp((-tau**2 * om**2)/(8*np.log(2)))

def delay(freq, shifts, kernel, T0, vg=0.7):
    times = np.empty([len(shifts)], dtype=np.float32)
    fid = np.empty([len(shifts)], dtype=np.float32)
    i = 0

    _, _in =  convolution(freq, np.ones_like(kernel), gauss_pulse(freq, T0), vg)
    for shift in shifts:

        t,f = convolution(freq, kernel, gauss_pulse(freq-shift, T0), vg)
        times[i] = (sum(t*abs(f)**2)/sum(abs(f)**2))


        fid[i] = np.sqrt( sum(abs(f)**2) / sum(abs(_in)**2) )
        i+=1

    return np.asarray(fid), np.asarray(times)


def efficiency(gamma, t, OutPulse):
    SquaredPulse = (abs(OutPulse))**2
    zero = np.argmin(t**2)

    return simps(SquaredPulse[zero:], t[zero:],even='avg') * gamma

def efficiency_map(sc_freq, sc_amp, times, shifts):
    effy = np.empty([len(times), len(shifts)],dtype=np.float32)
    for i in range(len(times)):
        for j in range(len(shifts)):
            time = times[i]
            shift = shifts[j]
            fourier_t, fourier_amp = convolution(sc_freq, sc_amp, pulse(sc_freq + shift, time), vg=0.7)
            effy[i,j] = efficiency(time, fourier_t, fourier_amp)
    return effy

def unitaryTransform(S):
    U = 1 / np.sqrt(2) * np.asarray([[1., 1.], [1., -1.]])
    return np.matmul(U, np.matmul(S, np.transpose(U)))

class ideal_spectra:
    def __init__(self, omega, gamma, rabi, rabidt, gamma_parasite=0):
        self.om = omega
        self.gm = gamma
        self.rb = rabi
        self.rbdt = rabidt
        self.gmp = gamma_parasite

    def reflection(self):
        return -0.5j * self.gm / (self.om - self.rb**2 / (4*(self.om-self.rbdt)) + 0.5j * (self.gm + self.gmp))

    def transmission(self):
        return 1 + self.reflection()

    def double_entry(self):
        return 1 + 2*self.reflection()