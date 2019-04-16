import one_D_scattering as ods
import numpy as np
import matplotlib.pyplot as plt
noa = 500

args = {
            'nat':noa, #number of atoms
            'nb':0, #number of neighbours in raman chanel (for L-atom only)
            's':'peculiar', #Stands for atom positioning : chain, nocorrchain and doublechain
            'dist':0.,  # sigma for displacement (choose 'chain' for gauss displacement., \lambda/2 units)
            'd' : 1.5, # distance from fiber
            'l0': 2./2, # mean distance between atoms (in lambda_m /2 units)
            'deltaP':ods.freq,  # array of freq
            'typ':'L',  # L or V for Lambda and V atom resp.
            'ff': 0.3 #filling factor (for ff_chain only)
            }

dtun = 10
foldername = 'comparingBEC'
separator = ','

PAIRS = False


nrandoms = 1
nof = len(ods.freq)
ref_raw = np.empty([nof, nrandoms], dtype = np.float32)
tran_raw = np.empty([nof, nrandoms], dtype=np.float32)

ods.RADIATION_MODES_MODEL = 1
for i in range(nrandoms):
    ods.SEED = i
    SE0 = ods.ensemble(**args)
    SE0.L = 2 * np.pi
    SE0.generate_ensemble()
    print('With full vacuum, Epoch %d / ' % (i + 1), nrandoms)

    ref_raw[:, i] = abs(SE0.Reflection)**2
    tran_raw[:, i] = abs(SE0.Transmittance)**2

ref_av = np.average(ref_raw, axis=1)
tran_av = np.average(tran_raw, axis=1)

toCsv = np.column_stack((ods.freq, ref_av, tran_av))
np.savetxt(foldername + '/NPeculiar2=%d' % noa + 'dd.csv', toCsv, delimiter=separator,
           fmt='%1.8f')

ods.DDI = 0
ods.RADIATION_MODES_MODEL = 1

for i in range(nrandoms):
    ods.SEED = i
    SE0 = ods.ensemble(**args)
    SE0.L = 2 * np.pi
    SE0.generate_ensemble()
    print('Without DDI, Epoch %d / ' % (i + 1), nrandoms)

    ref_raw[:, i] = abs(SE0.Reflection)**2
    tran_raw[:, i] = abs(SE0.Transmittance)**2

ref_av_nodd = np.average(ref_raw, axis=1)
tran_av_nodd = np.average(tran_raw, axis=1)

toCsv = np.column_stack((ods.freq, ref_av_nodd, tran_av_nodd))
np.savetxt(foldername + '/NPeculiar2=%d' % noa + 'nodd.csv', toCsv, delimiter=separator,
           fmt='%1.8f')

ods.RADIATION_MODES_MODEL = 0
for i in range(nrandoms):
    ods.SEED = i
    SE0 = ods.ensemble(**args)
    SE0.L = 2 * np.pi
    SE0.generate_ensemble()
    print('Without vacuum, Epoch %d / ' % (i + 1), nrandoms)

    ref_raw[:, i] = abs(SE0.Reflection)**2
    tran_raw[:, i] = abs(SE0.Transmittance)**2

ref_av_novac = np.average(ref_raw, axis=1)
tran_av_novac = np.average(tran_raw, axis=1)


toCsv = np.column_stack((ods.freq, ref_av_novac, tran_av_novac))
np.savetxt(foldername + '/NPeculiar2=%d' % noa + 'novac.csv', toCsv, delimiter=separator,
           fmt='%1.8f')

np.savetxt(foldername + '/NPeculiar2=%d' % noa + 'x0.csv', SE0.z, delimiter=separator,
           fmt='%1.8f')

plt.plot(ods.freq, ref_av)
plt.plot(ods.freq, tran_av)

plt.plot(ods.freq, ref_av_novac)
plt.plot(ods.freq, tran_av_novac)

plt.plot(ods.freq, ref_av_nodd)
plt.plot(ods.freq, tran_av_nodd)

plt.legend()
plt.show()
