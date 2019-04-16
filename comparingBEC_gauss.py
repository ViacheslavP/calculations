import one_D_scattering as ods
import matplotlib.pyplot as plt
import numpy as np

from one_D_scattering import freq
noa = 100
args = {
            'nat':noa, #number of atoms
            'nb':0, #number of neighbours in raman chanel (for L-atom only)
            's':'chain', #Stands for atom positioning : chain, nocorrchain and doublechain
            'dist':0.,  # sigma for displacement (choose 'chain' for gauss displacement., \lambda/2 units)
            'd' : 1.5, # distance from fiber
            'l0': 2./2, # mean distance between atoms (in lambda_m /2 units)
            'deltaP':freq,  # array of freq
            'typ':'L',  # L or V for Lambda and V atom resp.
            'ff': 0.3 #filling factor (for ff_chain only)
            }

foldername = 'comparingBEC'
sigmas = [0.5, 0.2, 0.1, 0.]
DDI =[1]
nrandoms = 100
nof = len(freq)

for ddi in DDI:
    ods.RADIATION_MODES_MODEL = ddi
    for sigma in sigmas:
        if sigma == 0.:
            anrandoms = 1
        else:
            anrandoms = nrandoms
        args['dist'] = sigma
        ref_raw = np.empty([nof, anrandoms], dtype=np.float32)
        tran_raw = np.empty([nof, anrandoms], dtype=np.float32)
        for i in range(anrandoms):
            ods.SEED = i
            SE0 = ods.ensemble(**args)
            SE0.L = 2 * np.pi
            SE0.generate_ensemble()
            if ddi:
                print('Gauss With DDI, Epoch %d / ' % (i + 1), nrandoms, 'sigma=', sigma, '  done!')
            else:
                print('Gauss Without DDI, Epoch %d / ' % (i + 1), nrandoms, 'sigma=', sigma, '  done!')

            ref_raw[:, i] = abs(SE0.Reflection) ** 2
            tran_raw[:, i] = abs(SE0.Transmittance) ** 2

        ref_av = np.average(ref_raw, axis=1)
        tran_av = np.average(tran_raw, axis=1)

        #saving
        separator = ','
        toCsv = np.column_stack((freq, ref_av, tran_av))
        np.savetxt(foldername + '/NGauss=%d' % noa + 'sigma=%s.csv' % sigma, toCsv, delimiter=separator,
                   fmt='%1.8f')

        fig = plt.figure(ddi+1)
        string = 'ddi='+str(ddi)+'sigma='+str(0.5*sigma)
        plt.plot(freq, ref_av, label=string+'ref')
        plt.plot(freq, tran_av, label=string+'trans')



plt.legend()
plt.show()




