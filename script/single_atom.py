import numpy as np
import one_D_scattering as ods
from matplotlib import pyplot as plt
#import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec

#from matplotlib import rc
#rc('font',**{'family':'serif','sans-serif':['Helvetica'],'size':16})
#rc('text', usetex=True)




"""
___________________________________________________

--------------------Decay--------------------------
___________________________________________________

from bmode import exact_mode

a = 2 * np.pi * 200 / 780
ndots = 41
c = 1
kv = 1
d00 = 0.5

argmode = {'k': 1,
        'n': 1.452,
        'a': a
        }
m = exact_mode(**argmode)

x = np.linspace(1, 5, ndots) * a

em = -m.E(x)
ez = m.Ez(x)
ep = m.dE(x)


gammaFull = np.zeros([ndots], dtype=complex)
for i in range(ndots):
    emfi = np.array([ep[i], ez[i], em[i]], dtype=complex)
    emfic = np.conjugate(np.array([ep[i], ez[i], em[i]], dtype=complex))

    gammaFull[i] = 1 + 4 * d00 * d00 * np.real(
    (np.dot(emfi, emfic) * 2 * np.pi * kv * (1 / m.vg - 1 / c)))


gammaPara = np.zeros([ndots], dtype=complex)
for i in range(ndots):
    emfi = np.array([ep[i], ez[i], em[i]], dtype=complex)
    emfic = np.conjugate(np.array([ep[i], ez[i], em[i]], dtype=complex))

    gammaPara[i] = 1 + 4 * d00 * d00 * np.real(
    (np.dot(emfi, emfic) * 2 * np.pi * kv * (1 / m.vg - 1 / c) + (ez[i] * ez[i] + ep[i]*np.conjugate(ep[i])) * 2 * np.pi * kv / c))


r00 = np.loadtxt('r00.txt', dtype=float, delimiter = ',')
f00 = np.loadtxt('f00.txt', dtype=float, delimiter = ',')
gammaExact = r00+f00
gammaFundamental = f00

gammaIntegral = np.zeros([ndots], dtype=complex)






import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec


ylim = [0.95, 1.6]
ylim2 = [0.0, 0.38]
ylimratio = (ylim[1] - ylim[0]) / (ylim2[1] - ylim2[0] + ylim[1] - ylim[0])
ylim2ratio = (ylim2[1] - ylim2[0]) / (ylim2[1] - ylim2[0] + ylim[1] - ylim[0])
gs = gridspec.GridSpec(2, 1, height_ratios=[ylimratio, ylim2ratio])
#fig = plt.figure()
#ax = fig.add_subplot(gs[0])
#ax2 = fig.add_subplot(gs[1])


#ax.plot(x/a, gammaFull,'b--', lw=1.5, label = 'Full subtraction')
#ax.plot(x/a, gammaPara,'r', lw=1.5, label = 'Paraxial subtraction')
#ax.plot(x/a, gammaExact,'m-.', lw=1.5, label = 'Exact')
#ax.plot(x/a, gammaIntegral, 'k', lw=1.5, label = 'Integral')
#ax.legend()

#ax2.plot(x/a, gammaFundamental,'g',lw=1.5,label = 'Fundamental mode only')
#ax2.legend()

#ax.set_xlim([1,3])
#ax2.set_xlim([1,3])

#ax.set_ylim(ylim)
#ax2.set_ylim(ylim2)
#plt.subplots_adjust(hspace=0.03)
#ax.spines['bottom'].set_visible(False)
#ax2.spines['top'].set_visible(False)
#ax.xaxis.tick_top()
#ax.tick_params(labeltop='off')
#ax2.xaxis.tick_bottom()
#ax2.set_xlabel('$r, a$')
#ax2.set_ylabel('$ {\gamma} / {\gamma_0} $')
#ax2.yaxis.set_label_coords(0.05, 0.5, transform=fig.transFigure)

#kwargs = dict(color='k', clip_on=False)
#xlim = ax.get_xlim()
#dx = .02 * (xlim[1] - xlim[0])
#dy = .01 * (ylim[1] - ylim[0]) / ylimratio
#ax.plot((xlim[0] - dx, xlim[0] + dx), (ylim[0] - dy, ylim[0] + dy), **kwargs)
#ax.plot((xlim[1] - dx, xlim[1] + dx), (ylim[0] - dy, ylim[0] + dy), **kwargs)
#dy = .01 * (ylim2[1] - ylim2[0]) / ylim2ratio
#ax2.plot((xlim[0] - dx, xlim[0] + dx), (ylim2[1] - dy, ylim2[1] + dy), **kwargs)
#ax2.plot((xlim[1] - dx, xlim[1] + dx), (ylim2[1] - dy, ylim2[1] + dy), **kwargs)
#ax.set_xlim(xlim)
#ax2.set_xlim(xlim)

#plt.show()


#plt.title('Spontaneous decay of ${}^{87}Rb$ atom near nanofiber')
#plt.ylabel('$\gamma / \gamma_0$')
#plt.xlabel('$r, a$')

plt.plot(x/a, gammaExact, 'r')
plt.plot(x/a, gammaFull, 'k')
plt.show()


np.savetxt('x.txt', x/a, newline='`,',fmt='%1.4f')
np.savetxt('gammaFull.txt', np.real(gammaFull), newline='`,',fmt='%1.4f')
np.savetxt('gammaPara.txt', np.real(gammaPara), newline='`,',fmt='%1.4f')
np.savetxt('gammaExact.txt', gammaExact, newline='`,',fmt='%1.4f')
np.savetxt('gammaFundamentalOnly.txt', 1+gammaFundamental, newline='`,',fmt='%1.4f')
"""
"""
___________________________________________________

------------------Scattering-----------------------
___________________________________________________
"""
freq = np.linspace(-0.25,0.25,450)

args10 = {

            'nat':1, #number of atoms
            'nb':0, #number of neighbours in raman chanel (for L-atom only)
            's':'chain', #Stands for atom positioning : chain, nocorrchain and doublechain
            'dist':0.,  # sigma for displacement (choose 'chain' for gauss displacement.)
            'd' : 2.0, # distance from fiber
            'l0':1.00, # mean distance between atoms (in lambda_m /2 units)
            'deltaP':freq,  # array of freq.
            'typ':'L',  # L or V for Lambda and V atom resp.
            'ff': 0.3

        }


args05 = {

            'nat':1, #number of atoms
            'nb':0, #number of neighbours in raman chanel (for L-atom only)
            's':'chain', #Stands for atom positioning : chain, nocorrchain and doublechain
            'dist':0.,  # sigma for displacement (choose 'chain' for gauss displacement.)
            'd' : 1.5, # distance from fiber
            'l0':1.00, # mean distance between atoms (in lambda_m /2 units)
            'deltaP':freq,  # array of freq.
            'typ':'L',  # L or V for Lambda and V atom resp.
            'ff': 0.3

        }

"""
_______________Fundamental modes only_____________
"""

ods.RADIATION_MODES_MODEL = 0 # = 1 iff assuming our model of radiation modes =0 else
ods.VACUUM_DECAY = 0 # = 0 iff assuming only decay into fundamental mode, =1 iff decay into fundamental and into radiation
ods.PARAXIAL = 1 # = 0 iff paraxial, =1 iff full mode
ods.SINGLE_RAMAN = True



sa = ods.ensemble(**args05)
sa.generate_ensemble()

T_fmo = sa.fullTransmittance
R_fmo = sa.fullReflection



np.savetxt('txts/freq.txt', freq, newline='`,',fmt='%1.5f')
np.savetxt('txts/'+'2.0'+'R_fmo.txt', R_fmo, newline='`,',fmt='%1.8f')
np.savetxt('txts/'+'2.0'+'T_fmo.txt', T_fmo, newline='`,',fmt='%1.8f')


plt.plot(freq, T_fmo, 'b', lw=1.25)
plt.plot(freq, R_fmo, 'r', lw=1.25)
plt.show()

ods.RADIATION_MODES_MODEL = 0 # = 1 iff assuming our model of radiation modes =0 else
ods.VACUUM_DECAY = 0 # = 0 iff assuming only decay into fundamental mode, =1 iff decay into fundamental and into radiation
ods.PARAXIAL = 1 # = 0 iff paraxial, =1 iff full mode
ods.SINGLE_RAMAN = True



sa = ods.ensemble(**args10)
sa.generate_ensemble()

T_fmo = sa.fullTransmittance
R_fmo = sa.fullReflection

np.savetxt('txts/'+'1.5'+'R_fmo.txt', R_fmo, newline='`,',fmt='%1.8f')
np.savetxt('txts/'+'1.5'+'T_fmo.txt', T_fmo, newline='`,',fmt='%1.8f')





plt.plot(freq, T_fmo, 'b', lw=1.25)
plt.plot(freq, R_fmo, 'r', lw=1.25)
plt.show()

print('Done')
"""
_______________Full subtraction____________________
"""

ods.RADIATION_MODES_MODEL = 1 # = 1 iff assuming our model of radiation modes =0 else
ods.VACUUM_DECAY = 1 # = 0 iff assuming only decay into fundamental mode, =1 iff decay into fundamental and into radiation
ods.PARAXIAL = 1 # = 0 iff paraxial, =1 iff full mode
ods.VERIFICATION_LAMBDA = 1
ods.FIRST = 0

sa = ods.ensemble(**args05)
sa.generate_ensemble()

T_05 = sa.fullTransmittance
R_05 = sa.fullReflection

sa = ods.ensemble(**args10)
sa.generate_ensemble()

T_10 = sa.fullTransmittance
R_10 = sa.fullReflection


plt.plot(freq, T_05, 'b', lw=1.25)
plt.plot(freq, T_10, 'r', lw=1.25)
plt.show()

plt.plot(freq, R_05, 'b', lw=1.25)
plt.plot(freq, R_10, 'r', lw=1.25)
plt.show()



"""
_____________Paraxial subtraction__________________
"""

#ods.RADIATION_MODES_MODEL = 1 # = 1 iff assuming our model of radiation modes =0 else
#ods.VACUUM_DECAY = 1 # = 0 iff assuming only decay into fundamental mode, =1 iff decay into fundamental and into radiation
#ods.PARAXIAL = 1 # = 0 iff paraxial, =1 iff full mode
#ods.VERIFICATION_LAMBDA = 1
#ods.FIRST = 1
#ods.SECOND = 1

#sa = ods.ensemble(**args)
#sa.generate_ensemble()

#T_ps = sa.fullTransmittance
#R_ps = sa.fullReflection

#plt.plot(freq, R_fmo,  lw=0.75,label='Reflection,\n decay in waveguide only')
#plt.plot(freq, T_fmo, lw=0.75, label='Transmittance,\n decay in waveguide only')

#plt.plot(freq, R_ps,'m--', label='Reflection,\n paraxial subtraction model')
#plt.plot(freq, T_ps,'--',color = '#ff5500ff', label='Transmittance,\n paraxial subtraction model')
#plt.plot(freq, 1 - R_ps - T_ps,'-.',color = '#aa55ffff', label='Loss,\n paraxial subtraction model')
#plt.legend()

#plt.xlabel('Detuning, $\gamma$')
#plt.show()

#plt.plot(freq, R_fmo,  lw=0.75,label='Reflection,\n decay in waveguide only')
#plt.plot(freq, R_ps,'m--', label='Reflection,\n paraxial subtraction model')
#plt.plot(freq, 1 - R_ps - T_ps,'-.',color = '#aa55ffff', label='Loss,\n paraxial subtraction model')
#plt.xlim([-0.25,0.25])
#plt.ylim([0,0.05])
#plt.show()


#np.savetxt('txts/'+'2.0'+'R_exact.txt', R_fs, newline='`,',fmt='%1.5f')
#np.savetxt('txts/'+'2.0'+'T_exact.txt', T_fs, newline='`,',fmt='%1.5f')
#np.savetxt('txts/'+'2.0'+'Loss.txt', 1-R_fs-T_fs, newline='`,',fmt='%1.5f')



#Full Transmittance:

#ax[0].plot(freq, sa.fullTransmittance, 'r')
#ax[0].set_yticks(np.arange(0.5, 1, 0.125))
#ax[0].set_ylabel('Transmission')

#FullReflection:

#ax[1].plot(freq, sa.fullReflection, 'b')
#ax[1].set_ylabel('Reflectance')
#ax[1].set_yticks(np.arange(0.0, 0.5, 0.125))
#ax[1].set_xlabel('Detuning, $\gamma$')
"""
ax[0].text(-0.125, 0.5, 'Transmittance',
        horizontalalignment='right',
        verticalalignment='center',
        rotation='vertical',
        transform=ax[0].transAxes)

ax[1].text(-0.125, 0.5, 'Reflection',
        horizontalalignment='right',
        verticalalignment='center',
        rotation='vertical',
        transform=ax[1].transAxes)
        """
#plt.show()
#plt.clr()

