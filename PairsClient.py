import sys
import os
import core.one_D_scattering as ods
from core.wave_pack import convolution, delay
from core.wave_pack import gauss_pulse_v2 as pulse
import numpy as np
import paramiko
sq_reduce = lambda A: np.add.reduce(np.square(np.absolute(A)), axis=1)


CSVPATH = 'data/m_arrays/wpairs/'
if not os.path.exists(CSVPATH):
    os.makedirs(CSVPATH)

SERVERPATH =
def toMathematica(filename, *argv):
    toCsv = np.column_stack(argv)
    np.savetxt(CSVPATH + filename + '.csv', toCsv, delimiter=',', fmt='%1.8f')

def connectMSU():
    import getpass
    host = 'cluster.qotlabs.org'
    user = 'pivovarovva'
    port = 22
    secret = getpass.getpass('Password for '+user+':')

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, username=user, password=secret, port=port)

    client.close()


#filenames
filenames = []
for num in ('ten', 'hundred'):
    for rt in ('', 'AT'):
        for cor in ('', 'nc'):
            filenames.append(num+rt+cor)
for filename in ()
    np.load('data/server/')
    #Transmittance
    _, ft = convolution(freq, elasticSc.Transmittance, pulse(freq + SHIFT, pdTime))
    _, fr = convolution(freq, elasticSc.Reflection, pulse(freq + SHIFT, pdTime))
    ti, fi = convolution(freq, np.ones_like(freq), pulse(freq + SHIFT, pdTime))