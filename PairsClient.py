import sys
import os
import core.one_D_scattering as ods
from core.wave_pack import convolution, delay
from core.wave_pack import inverse_pulse as pulse
from core.wave_pack import efficiency_map, unitaryTransform, efficiency_sq
import numpy as np
import paramiko
sq_reduce = lambda A: np.add.reduce(np.square(np.absolute(A)), axis=0)


CSVPATHWP = 'data/m_arrays/wpairs_09.08.2019/'
CSVPATHNC = 'data/m_arrays/nocorr_09.08.2019/'

for CSVPATH in (CSVPATHWP, CSVPATHNC):
    if not os.path.exists(CSVPATH):
        raise ValueError

def toMathematica(filename, *argv):
    toCsv = np.column_stack(argv)
    np.savetxt(NEWPATH + '/' + filename + '.csv', toCsv, delimiter=',', fmt='%1.8f')

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

for CSVPATH in (CSVPATHNC, CSVPATHWP):
    for filename in filenames:
        ensemble = np.load(CSVPATH+filename)
        ensemble_inv = np.load(CSVPATH+'inv'+filename)
        if 'ten' in filename:
            nat = 10
        elif 'hundred' in filename:
            nat = 100
        else:
            raise ValueError


        freq = ensemble.f.freq


        # Find optimal
        Smatrix = np.moveaxis( np.asarray([[ensemble.f.SF1_pm, ensemble.f.SB1_pm],
                                [ensemble_inv.f.SF1_pm, ensemble_inv.f.SB1_pm]]), -1,0)
        Sunit = unitaryTransform(Smatrix)


        TS = efficiency_map(freq, Sunit[:, 0, 0], pdGammas, pdShifts)
        print(Sunit[:, 0,0])
        maxTS = np.unravel_index(TS.argmax(), TS.shape)
        gamma = pdGammas[maxTS[0]]; shift = pdShifts[maxTS[1]]

        def opt_conv(x):
            return convolution(freq, x, pulse(freq + shift, gamma))

        time, fi = opt_conv(np.ones_like(ensemble.f.SF1_pm))

        #Forward Intensity

        _, ft = opt_conv(ensemble.f.SF1_pm)
        _, ft_i = opt_conv(ensemble.f.SF1_pp)

        tf0m = np.empty((nat, len(ft)), dtype=np.complex)
        tf0p = np.empty((nat, len(ft)), dtype=np.complex)
        tfmm = np.empty((nat, len(ft)), dtype=np.complex)
        tfmp = np.empty((nat, len(ft)), dtype=np.complex)


        for i in range(nat):
            _, tf0m[i, :] = opt_conv(1j*ensemble.f.SF2_0m[i,:])
            _, tf0p[i, :] = opt_conv(1j*ensemble.f.SF2_0p[i,:])
            _, tfmm[i, :] = opt_conv(1j*ensemble.f.SF2_mm[i,:])
            _, tfmp[i, :] = opt_conv(1j*ensemble.f.SF2_mp[i,:])

        fullTransmittance =   abs(ensemble.f.SF1_pm)**2 \
                            + abs(ensemble.f.SF1_pp)**2 \
                            + sq_reduce(ensemble.f.SF2_0m) \
                            + sq_reduce(ensemble.f.SF2_0p) \
                            + sq_reduce(ensemble.f.SF2_mm) \
                            + sq_reduce(ensemble.f.SF2_mp)

        fullReflectance   =   abs(ensemble.f.SB1_pm)**2 \
                            + abs(ensemble.f.SB1_pp)**2 \
                            + sq_reduce(ensemble.f.SB2_0m) \
                            + sq_reduce(ensemble.f.SB2_0p) \
                            + sq_reduce(ensemble.f.SB2_mm) \
                            + sq_reduce(ensemble.f.SB2_mp)

        Forward = abs(ft)**2 \
                + abs(ft_i)**2 \
                + sq_reduce(tf0m) \
                + sq_reduce(tf0p) \
                + sq_reduce(tfmm) \
                + sq_reduce(tfmp)

        # Backward Intensity

        _, fr = opt_conv(ensemble.f.SB1_pm)
        _, fr_i = opt_conv(ensemble.f.SB1_pp)

        tb0m = np.empty((nat, len(ft)), dtype=np.complex)
        tb0p = np.empty((nat, len(ft)), dtype=np.complex)
        tbmm = np.empty((nat, len(ft)), dtype=np.complex)
        tbmp = np.empty((nat, len(ft)), dtype=np.complex)

        for i in range(nat):
            _, tb0m[i, :] = opt_conv(-1j*ensemble.f.SB2_0m[i, :])
            _, tb0p[i, :] = opt_conv(-1j*ensemble.f.SB2_0p[i, :])
            _, tbmm[i, :] = opt_conv(-1j*ensemble.f.SB2_mm[i, :])
            _, tbmp[i, :] = opt_conv(-1j*ensemble.f.SB2_mp[i, :])

        Backward = abs(fr) ** 2 \
                  + abs(fr_i) ** 2 \
                  + sq_reduce(tb0m) \
                  + sq_reduce(tb0p) \
                  + sq_reduce(tbmm) \
                  + sq_reduce(tbmp)

        SingleEntry = Forward + Backward
        SingleEntryElastic = abs(ft)**2 + abs(fr)**2




        #Sagnac amplitudes: Left port (Same one)

        sz_pm = (ensemble.f.SF1_pm + ensemble_inv.f.SF1_pm)/2 + (ensemble.f.SB1_pm + ensemble_inv.f.SB1_pm)/2
        sz_pp = (ensemble.f.SF1_pp + ensemble_inv.f.SF1_pp)/2 + (ensemble.f.SB1_pp + ensemble_inv.f.SB1_pp)/2
        sz_mm = (ensemble.f.SF2_mm + ensemble_inv.f.SF2_mm)/2 + (ensemble.f.SB2_mm + ensemble_inv.f.SB2_mm)/2
        sz_mp = (ensemble.f.SF2_mp + ensemble_inv.f.SF2_mp)/2 + (ensemble.f.SB2_mp + ensemble_inv.f.SB2_mp)/2
        sz_0m = (ensemble.f.SF2_0m + ensemble_inv.f.SF2_0m)/2 + (ensemble.f.SB2_0m + ensemble_inv.f.SB2_0m)/2
        sz_0p = (ensemble.f.SF2_0p + ensemble_inv.f.SF2_0p)/2 + (ensemble.f.SB2_0p + ensemble_inv.f.SB2_0p)/2

        #Sagnac amplitudes: Right port (Another one)

        psz_pm = (ensemble.f.SF1_pm - ensemble_inv.f.SF1_pm)/2 + (ensemble.f.SB1_pm - ensemble_inv.f.SB1_pm)/2
        psz_pp = (ensemble.f.SF1_pp - ensemble_inv.f.SF1_pp)/2 + (ensemble.f.SB1_pp - ensemble_inv.f.SB1_pp)/2
        psz_mm = (ensemble.f.SF2_mm - ensemble_inv.f.SF2_mm)/2 + (ensemble.f.SB2_mm - ensemble_inv.f.SB2_mm)/2
        psz_mp = (ensemble.f.SF2_mp - ensemble_inv.f.SF2_mp)/2 + (ensemble.f.SB2_mp - ensemble_inv.f.SB2_mp)/2
        psz_0m = (ensemble.f.SF2_0m - ensemble_inv.f.SF2_0m)/2 + (ensemble.f.SB2_0m - ensemble_inv.f.SB2_0m)/2
        psz_0p = (ensemble.f.SF2_0p - ensemble_inv.f.SF2_0p)/2 + (ensemble.f.SB2_0p - ensemble_inv.f.SB2_0p)/2

        _, szpm = opt_conv(sz_pm)
        _, szpp = opt_conv(sz_pp)

        sz0m = np.empty((nat, len(ft)), dtype=np.complex)
        sz0p = np.empty((nat, len(ft)), dtype=np.complex)
        szmm = np.empty((nat, len(ft)), dtype=np.complex)
        szmp = np.empty((nat, len(ft)), dtype=np.complex)

        for i in range(nat):
            _, sz0m[i, :] = opt_conv(sz_0m[i, :])
            _, sz0p[i, :] = opt_conv(sz_0p[i, :])
            _, szmm[i, :] = opt_conv(sz_mm[i, :])
            _, szmp[i, :] = opt_conv(sz_mp[i, :])


        _, pszpm = opt_conv(psz_pm)
        _, pszpp = opt_conv(psz_pp)

        psz0m = np.empty((nat, len(ft)), dtype=np.complex)
        psz0p = np.empty((nat, len(ft)), dtype=np.complex)
        pszmm = np.empty((nat, len(ft)), dtype=np.complex)
        pszmp = np.empty((nat, len(ft)), dtype=np.complex)

        for i in range(nat):
            _, psz0m[i, :] = opt_conv(psz_0m[i, :])
            _, psz0p[i, :] = opt_conv(psz_0p[i, :])
            _, pszmm[i, :] = opt_conv(psz_mm[i, :])
            _, pszmp[i, :] = opt_conv(psz_mp[i, :])

        fullSagnac   =   abs(sz_pm)**2 \
                            + abs(sz_pp)**2 \
                            + sq_reduce(sz_0m) \
                            + sq_reduce(sz_0p) \
                            + sq_reduce(sz_mm) \
                            + sq_reduce(sz_mp)

        pfullSagnac   =   abs(psz_pm)**2 \
                            + abs(psz_pp)**2 \
                            + sq_reduce(psz_0m) \
                            + sq_reduce(psz_0p) \
                            + sq_reduce(psz_mm) \
                            + sq_reduce(psz_mp)

        Left =      abs(szpm) ** 2 \
                  + abs(szpp) ** 2 \
                  + sq_reduce(sz0m) \
                  + sq_reduce(sz0p) \
                  + sq_reduce(szmm) \
                  + sq_reduce(szmp)

        Right =     abs(pszpm) ** 2 \
                  + abs(pszpp) ** 2 \
                  + sq_reduce(psz0m) \
                  + sq_reduce(psz0p) \
                  + sq_reduce(pszmm) \
                  + sq_reduce(pszmp)

        from matplotlib import pyplot as plt




        if 'ten' in filename:
            NEWPATH = 'data/m_arrays/tens'
        elif 'hundred' in filename:
            NEWPATH = 'data/m_arrays/hundreds'
        else:
            raise ValueError

        if not os.path.exists(NEWPATH):
            os.makedirs(NEWPATH)

        if 'nocorr' in CSVPATH:
            typ = 'array'
        elif 'pair' in CSVPATH:
            typ = 'chain'
        else:
            raise ValueError

        if 'AT' in filename:
            pref = 'cf'
        else:
            pref = ''


        toMathematica(typ + 'OS'  + pref,
                                        freq,
                                        fullTransmittance,
                                        fullReflectance,
                                        1 - fullReflectance - fullTransmittance,
                                        time,
                                        abs(fi)**2,
                                        Forward,
                                        Backward)

        toMathematica(typ + 'TS'  + pref,
                                        freq,
                                        fullSagnac,
                                        pfullSagnac,
                                        1 - fullSagnac - pfullSagnac,
                                        time,
                                        Left,
                                        Right)
                                        



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

    sz = (ensemble.f.Transmittance + ensemble_inv.f.Transmittance +
          ensemble.f.Reflection + ensemble_inv.f.Reflection)/2

    sz_i = (ensemble.f.iTransmittance + ensemble_inv.f.iTransmittance +
          ensemble.f.iReflection + ensemble_inv.f.iReflection)/2

    sz_mm = -1j*(ensemble.f.TF2_mm + ensemble_inv.f.TF2_mm -
                 ensemble.f.TB2_mm - ensemble_inv.f.TB2_mm)

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

    from matplotlib import pyplot as plt

    plt.plot(freq, fullSagnac)
    plt.plot(freq, pfullSagnac)
    plt.show()

    #toMathematica(filename+'spectra', freq, fR, fT, fRram, fTram, fullSagnac, SagnacRam, pfullSagnac, pSagnacRam)
"""