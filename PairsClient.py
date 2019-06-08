import sys
import os
import core.one_D_scattering as ods
from core.wave_pack import convolution, delay
from core.wave_pack import gauss_pulse_v2 as pulse
import numpy as np
sq_reduce = lambda A: np.add.reduce(np.square(np.absolute(A)), axis=1)


CSVPATH =
def toMathematica(filename, *argv):
    toCsv = np.column_stack(argv)
    np.savetxt(CSVPATH + filename + '.csv', toCsv, delimiter=',', fmt='%1.8f')

for filename in ()
    #Transmittance
    _, ft = convolution(freq, elasticSc.Transmittance, pulse(freq + SHIFT, pdTime))
    _, fr = convolution(freq, elasticSc.Reflection, pulse(freq + SHIFT, pdTime))
    ti, fi = convolution(freq, np.ones_like(freq), pulse(freq + SHIFT, pdTime))