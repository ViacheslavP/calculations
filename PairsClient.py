import sys
import os
import core.one_D_scattering as ods
from core.wave_pack import convolution, delay
from core.wave_pack import gauss_pulse_v2 as pulse
from core.wave_pack import efficiency_map, unitaryTransform
import numpy as np
import paramiko
sq_reduce = lambda A: np.add.reduce(np.square(np.absolute(A)), axis=0)


CSVPATH = 'data/m_arrays/wpairs/'
if not os.path.exists(CSVPATH):
    os.makedirs(CSVPATH)
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
        filenames.append(num+rt+'.npz')
fullfilenames = filenames + ['inv'+x for x in filenames]


#download_files(fullfilenames)
atfnames = list(filter(lambda x: 'AT' in x, filenames))


#Optimization parameters
nshifts = 100
ntimes = 200

pdGammas = np.linspace(0.02, 1, ntimes)
pdShifts = np.linspace(3, 5, nshifts)


for filename in atfnames:
    ensemble = np.load(CSVPATH+filename)
    ensemble_inv = np.load(CSVPATH+filename)
    if 'ten' in filename:
        nat = 10
    elif 'hundred' in filename:
        nat = 100
    else:
        raise ValueError


    freq = ensemble.f.freq


    # Find optimal
    Smatrix = np.moveaxis( np.asarray([[ensemble.f.Transmittance, ensemble.f.Reflection],
                            [ensemble_inv.f.Reflection, ensemble_inv.f.Transmittance]]), -1,0)
    Sunit = unitaryTransform(Smatrix)


    TS = efficiency_map(freq, Sunit[:, 0, 0], pdGammas, pdShifts)
    maxTS = np.unravel_index(TS.argmax(), TS.shape)
    gamma = pdGammas[maxTS[0]]; shift = pdShifts[maxTS[1]]

    opt_conv = lambda x: convolution(freq, x, pulse(freq + shift, gamma))

    time, fr = opt_conv(np.ones_like(ensemble.f.Transmittance))
    #Forward Intensity

    _, ft = opt_conv(ensemble.f.Transmittance)
    _, ft_i = opt_conv(ensemble.f.iTransmittance)

    tf0m = np.empty((nat, len(ft)), dtype=np.complex)
    tf0p = np.empty((nat, len(ft)), dtype=np.complex)
    tfmm = np.empty((nat, len(ft)), dtype=np.complex)
    tfmp = np.empty((nat, len(ft)), dtype=np.complex)


    for i in range(nat):
        _, tf0m[i, :] = opt_conv(1j*ensemble.f.TF2_0m[i,:])
        _, tf0p[i, :] = opt_conv(1j*ensemble.f.TF2_0p[i,:])
        _, tfmm[i, :] = opt_conv(1j*ensemble.f.TF2_mm[i,:])
        _, tfmp[i, :] = opt_conv(1j*ensemble.f.TF2_mp[i,:])

    Forward = abs(ft)**2 \
            + abs(ft_i)**2 \
            + sq_reduce(tf0m) \
            + sq_reduce(tf0p) \
            + sq_reduce(tfmm) \
            + sq_reduce(tfmp)

    # Backward Intensity

    _, fr = opt_conv(ensemble.f.Reflection)
    _, fr_i = opt_conv(ensemble.f.iReflection)

    tb0m = np.empty((nat, len(ft)), dtype=np.complex)
    tb0p = np.empty((nat, len(ft)), dtype=np.complex)
    tbmm = np.empty((nat, len(ft)), dtype=np.complex)
    tbmp = np.empty((nat, len(ft)), dtype=np.complex)

    for i in range(nat):
        _, tb0m[i, :] = opt_conv(-1j*ensemble.f.TB2_0m[i, :])
        _, tb0p[i, :] = opt_conv(-1j*ensemble.f.TB2_0p[i, :])
        _, tbmm[i, :] = opt_conv(-1j*ensemble.f.TB2_mm[i, :])
        _, tbmp[i, :] = opt_conv(-1j*ensemble.f.TB2_mp[i, :])

    Backward = abs(fr) ** 2 \
              + abs(fr_i) ** 2 \
              + sq_reduce(tb0m) \
              + sq_reduce(tb0p) \
              + sq_reduce(tbmm) \
              + sq_reduce(tbmp)

    SingleEntry = Forward + Backward



