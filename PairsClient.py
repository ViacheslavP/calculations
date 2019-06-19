import sys
import os
import core.one_D_scattering as ods
from core.wave_pack import convolution, delay
from core.wave_pack import inverse_pulse as pulse
from core.wave_pack import efficiency_map, unitaryTransform, efficiency_sq
import numpy as np
import paramiko
sq_reduce = lambda A: np.add.reduce(np.square(np.absolute(A)), axis=0)


CSVPATH = 'data/m_arrays/wpairs_nocorr/'
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

def plot_map(intimes, shifts, eff_map):
    from mpl_toolkits.mplot3d import Axes3D
    x, y = np.meshgrid(intimes, shifts)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x, y, np.transpose(eff_map))
    plt.show()



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

"""
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
    print(Sunit[:, 0,0])
    maxTS = np.unravel_index(TS.argmax(), TS.shape)
    gamma = pdGammas[maxTS[0]]; shift = pdShifts[maxTS[1]]

    def opt_conv(x):
        return convolution(freq, x, pulse(freq + shift, gamma))

    time, fi = opt_conv(np.ones_like(ensemble.f.Transmittance))

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
    SingleEntryElastic = abs(ft)**2 + abs(fr)**2

    #Sagnac intensity
    sz = ensemble.f.Transmittance + ensemble_inv.f.Reflection
    sz_i = ensemble.f.iTransmittance + ensemble_inv.f.iReflection
    sz_mm = -1j*(ensemble.f.TF2_mm - ensemble_inv.f.TB2_mm)
    sz_mp = -1j*(ensemble.f.TF2_mp - ensemble_inv.f.TB2_mp)
    sz_0m = -1j*(ensemble.f.TF2_0m - ensemble_inv.f.TB2_0m)
    sz_0p = -1j*(ensemble.f.TF2_0p - ensemble_inv.f.TB2_0p)


    _, fs = opt_conv(sz)
    _, fs_i = opt_conv(sz_i)

    sz0m = np.empty((nat, len(ft)), dtype=np.complex)
    sz0p = np.empty((nat, len(ft)), dtype=np.complex)
    szmm = np.empty((nat, len(ft)), dtype=np.complex)
    szmp = np.empty((nat, len(ft)), dtype=np.complex)

    for i in range(nat):
        _, sz0m[i, :] = opt_conv(sz_0m[i, :])
        _, sz0p[i, :] = opt_conv(sz_0p[i, :])
        _, szmm[i, :] = opt_conv(sz_mm[i, :])
        _, szmp[i, :] = opt_conv(sz_mp[i, :])

    Sagnac = abs(fs) ** 2 \
              + abs(fs_i) ** 2 \
              + sq_reduce(sz0m) \
              + sq_reduce(sz0p) \
              + sq_reduce(szmm) \
              + sq_reduce(szmp)

    SagnacElastic = abs(fs)**2

    print('Elastic eff:', efficiency_sq(gamma, time, Sagnac))
    print('Inelastic eff:', efficiency_sq(gamma, time, SagnacElastic))

    toMathematica(filename+'pulses', time,
                                    abs(fi)**2,
                                    SingleEntry,
                                    SingleEntryElastic,
                                    Sagnac,
                                    SagnacElastic)
"""
"""
for filename in filenames:

    ensemble = np.load(CSVPATH+filename)
    ensemble_inv = np.load(CSVPATH+filename)
    if 'ten' in filename:
        nat = 10
    elif 'hundred' in filename:
        nat = 100
    else:
        raise ValueError

    freq = ensemble.f.freq

    fR = ensemble.f.fullReflection
    fT = ensemble.f.fullTransmittance

    fRram = ensemble.f.fullReflection - abs(ensemble.f.Reflection)**2
    fTram = ensemble.f.fullTransmittance - abs(ensemble.f.Transmittance)**2

    sz = ensemble.f.Transmittance + ensemble_inv.f.Reflection
    sz_i = ensemble.f.iTransmittance + ensemble_inv.f.iReflection
    sz_mm = -1j*(ensemble.f.TF2_mm - ensemble_inv.f.TB2_mm)
    sz_mp = -1j*(ensemble.f.TF2_mp - ensemble_inv.f.TB2_mp)
    sz_0m = -1j*(ensemble.f.TF2_0m - ensemble_inv.f.TB2_0m)
    sz_0p = -1j*(ensemble.f.TF2_0p + ensemble_inv.f.TB2_0p)


    psz = ensemble.f.Transmittance - ensemble_inv.f.Reflection
    psz_i = ensemble.f.iTransmittance - ensemble_inv.f.iReflection
    psz_mm = -1j*(ensemble.f.TF2_mm + ensemble_inv.f.TB2_mm)
    psz_mp = -1j*(ensemble.f.TF2_mp + ensemble_inv.f.TB2_mp)
    psz_0m = -1j*(ensemble.f.TF2_0m + ensemble_inv.f.TB2_0m)
    psz_0p = -1j*(ensemble.f.TF2_0p - ensemble_inv.f.TB2_0p)


    print(np.amax(abs(sz_mm)**2), np.amax(abs(psz_mm)**2))
    print(np.amax(abs(sz_mp)**2), np.amax(abs(psz_mp)**2))
    print(np.amax(abs(sz_0p)**2), np.amax(abs(psz_0p)**2))
    print(np.amax(abs(sz_0m)**2), np.amax(abs(psz_0m)**2))
    print(np.amax(abs(sz_i) ** 2), np.amax(abs(psz_i) ** 2))

    SagnacSp = abs(sz)**2
    fullSagnac = abs(sz)**2 + abs(sz_i)**2 +  \
                 sq_reduce(sz_mm) + \
                 sq_reduce(sz_mp) + \
                 sq_reduce(sz_0m) + \
                 sq_reduce(sz_0p)
    SagnacRam = fullSagnac - SagnacSp

    pSagnacSp = abs(psz)**2
    pfullSagnac = abs(psz)**2 + abs(psz_i)**2 +  \
                 sq_reduce(psz_mm) + \
                 sq_reduce(psz_mp) + \
                 sq_reduce(psz_0m) + \
                 sq_reduce(psz_0p)
    pSagnacRam = pfullSagnac - pSagnacSp

    toMathematica(filename+'spectra', freq, fR, fT, fRram, fTram, fullSagnac, SagnacRam, pfullSagnac, pSagnacRam)
"""
