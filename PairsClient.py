import sys
import os
import core.one_D_scattering as ods
from core.wave_pack import convolution, delay
from core.wave_pack import gauss_pulse_v2 as pulse
from core.wave_pack import efficiency_map
import numpy as np
import paramiko
sq_reduce = lambda A: np.add.reduce(np.square(np.absolute(A)), axis=1)


CSVPATH = 'data/m_arrays/wpairs/'
if not os.path.exists(CSVPATH):
    os.makedirs(CSVPATH)

SERVERPATH = '/shared/data_m/mem_spectra/'
def toMathematica(filename, *argv):
    toCsv = np.column_stack(argv)
    np.savetxt(CSVPATH + filename + '.csv', toCsv, delimiter=',', fmt='%1.8f')

def download_files(fnames):
    import getpass
    host = 'cluster.qotlabs.org'
    user = 'pivovarovva'
    port = 22
    secret = getpass.getpass('Password for '+user+':')

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, username=user, password=secret, port=port)

    sftp = client.open_sftp()
    for fname in fnames:
        sftp.get(SERVERPATH+fname, CSVPATH+fname)
    client.close()




#List of filenames
filenames = []
for num in ('ten', 'hundred'):
    for rt in ('', 'AT'):
        for cor in ('', 'nc'):
            filenames.append(num+rt+cor)
fullfilenames = filenames + ['inv'+x for x in filenames]


download_files(fullfilenames)
atfnames = list(filter(lambda x: 'AT' in x, filenames))
for filename in atfnames:
    ensemble = np.load(CSVPATH+filename)
    #Find optimal
    freq = ensemble.f.freq


    Smatrix = np.moveaxis( np.asarray([[ensemble.f.Transmittance, ensemble.f.Reflection],
                            [ensemble.f.Reflection, ensemble.f.Transmittance]]), -1,0)
    TS_chain = efficiency_map(freq, uSu_chain[:, 0, 0], pdGammas, pdShifts)
    maxTS = np.unravel_index(effyTS_chain.argmax(), effyTS_chain.shape)
    gamma = pdGammas[maxTS[0]]; shift = pdShifts[maxTS[1]]
    #Transmittance
    freq = ensemble.freq
    _, ft = convolution(freq, ensemble.f.Transmittance, pulse(freq + SHIFT, pdTime))



    _, fr = convolution(freq, elasticSc.Reflection, pulse(freq + SHIFT, pdTime))
    ti, fi = convolution(freq, np.ones_like(freq), pulse(freq + SHIFT, pdTime))