import numpy as np
import one_D_scattering as ods
from matplotlib import pyplot as plt
#from matplotlib import rc
#rc('font',**{'family':'serif','sans-serif':['Helvetica'],'size':16})
#rc('text', usetex=True)




freq = np.linspace(-2.5, 2.5, 180)

#default arguments
args_def = {

    'nat': 5,  # number of atoms
    'nb': 4,  # number of neighbours in raman chanel (for L-atom only)
    's': 'chain',  # Stands for atom positioning : chain, nocorrchain and doublechain
    'dist': 0.,  # sigma for displacement (choose 'chain' for gauss displacement.)
    'd': 2.0,  # distance from fiber
    'l0': 1,  # mean distance between atoms (in lambda_m /2 units)
    'deltaP': freq,  # array of freq.
    'typ': 'L',  # L or V for Lambda and V atom resp.
    'ff': 0.3

}




"""
__________________________________________________
Raman vs Rayleigh
__________________________________________________
"""

"""
Order, d = 1
"""
args = args_def.copy()
args['d'] = 2.

ods.SINGLE_RAMAN = 0

SE0_order10 = ods.ensemble(**args)
SE0_order10.generate_ensemble()

ods.SINGLE_RAMAN = 1

SE1_order10 = ods.ensemble(**args)
SE1_order10.generate_ensemble()


Refl_Ray_10_order= SE0_order10.fullReflection
Refl_Raman_10_order = SE1_order10.fullReflection
Trans_Ray_10_order = SE0_order10.fullTransmittance
Trans_Raman_10_order = SE1_order10.fullTransmittance

"""
Disorder, d = 1
"""

args = args_def.copy()
args['d'] = 2.
args['s'] = 'nocorrchain'

ods.SINGLE_RAMAN = 0

SE0_disorder10 = ods.ensemble(**args)
SE0_disorder10.generate_ensemble()

ods.SINGLE_RAMAN = 1

SE1_disorder10 = ods.ensemble(**args)
SE1_disorder10.generate_ensemble()


Refl_Ray_10_disorder = SE0_disorder10.fullReflection
Refl_Raman_10_disorder = SE1_disorder10.fullReflection
Trans_Ray_10_disorder = SE0_disorder10.fullTransmittance
Trans_Raman_10_disorder = SE1_disorder10.fullTransmittance




"""
Order, d = 0.5
"""
args = args_def.copy()
args['d'] = 1.5

ods.SINGLE_RAMAN = 0

SE0_order05 = ods.ensemble(**args)
SE0_order05.generate_ensemble()

ods.SINGLE_RAMAN = 1

SE1_order05 = ods.ensemble(**args)
SE1_order05.generate_ensemble()


Refl_Ray_05_order= SE0_order05.fullReflection
Refl_Raman_05_order = SE1_order05.fullReflection
Trans_Ray_05_order = SE0_order05.fullTransmittance
Trans_Raman_05_order = SE1_order05.fullTransmittance

"""
Disorder, d = 0.5
"""
args = args_def.copy()
args['d'] = 1.5
args['s'] = 'nocorrchain'


ods.SINGLE_RAMAN = 0

SE0_disorder05 = ods.ensemble(**args)
SE0_disorder05.generate_ensemble()

ods.SINGLE_RAMAN = 1

SE1_disorder05 = ods.ensemble(**args)
SE1_disorder05.generate_ensemble()


Refl_Ray_05_disorder = SE0_disorder05.fullReflection
Refl_Raman_05_disorder = SE1_disorder05.fullReflection
Trans_Ray_05_disorder = SE0_disorder05.fullTransmittance
Trans_Raman_05_disorder = SE1_disorder05.fullTransmittance


plt.title('Order', fontsize=16)
plt.plot(freq, Trans_Ray_10_order, 'r--', label='d=1.0, Rayleigh, order ', lw=1.5)
plt.plot(freq, Trans_Raman_10_order, 'r-', label='d=1.0, Rayleigh+Single Raman, order', lw=1.5)
plt.plot(freq, Trans_Ray_05_order, 'b--', label='d=0.5, Rayleigh, order', lw=1.5)
plt.plot(freq, Trans_Raman_05_order, 'b-', label='d=0.5, Rayleigh+Single Raman, order', lw=1.5)
plt.legend()
plt.xlabel('Detuning, $\gamma$', fontsize=16)
plt.ylabel('Transmission', fontsize=16)
plt.show()
plt.clf()

plt.title('Disorder', fontsize=16)
plt.plot(freq, Trans_Ray_10_disorder, 'm--', label='d=1.0, Rayleigh , disorder', lw=1.5)
plt.plot(freq, Trans_Raman_10_disorder, 'm-', label='d=1.0, Rayleigh+Single Raman, disorder', lw=1.5)
plt.plot(freq, Trans_Ray_05_disorder, 'c--', label='d=0.5, Rayleigh, disorder', lw=1.5)
plt.plot(freq, Trans_Raman_05_disorder, 'c-', label='d=0.5, Rayleigh+Single Raman, disorder', lw=1.5)
plt.legend()
plt.xlabel('Detuning, $\gamma$', fontsize=16)
plt.ylabel('Transmission', fontsize=16)
plt.show()
plt.clf()

plt.title('Order', fontsize=16)
plt.plot(freq, Refl_Ray_10_order, 'r--', label='d=1.0, Rayleigh, order', lw=1.5)
plt.plot(freq, Refl_Raman_10_order, 'r-', label='d=1.0, Rayleigh+Single Raman, order', lw=1.5)
plt.plot(freq, Refl_Ray_05_order, 'b--', label='d=0.5, Rayleigh, order', lw=1.5)
plt.plot(freq, Refl_Raman_05_order, 'b-', label='d=0.5, Rayleigh+Single Raman, order', lw=1.5)
plt.legend()
plt.xlabel('Detuning, $\gamma$', fontsize=16)
plt.ylabel('Reflection', fontsize=16)
plt.show()
plt.clf()

plt.title('Disorder', fontsize=16)
plt.plot(freq, Refl_Ray_10_disorder, 'm--', label='d=1.0, Rayleigh, disorder', lw=1.5)
plt.plot(freq, Refl_Raman_10_disorder, 'm-', label='d=1.0, Rayleigh+Single Raman, disorder', lw=1.5)
plt.plot(freq, Refl_Ray_05_disorder, 'c--', label='d=0.5, Rayleigh, disorder', lw=1.5)
plt.plot(freq, Refl_Raman_05_disorder, 'c-', label='d=0.5, Rayleigh+Single Raman, disorder', lw=1.5)
plt.legend()
plt.xlabel('Detuning, $\gamma$', fontsize=16)
plt.ylabel('Reflection', fontsize=16)
plt.show()
plt.clf()



"""
Mathematica formatting
"""

np.savetxt('freq.txt', freq, newline='`,',fmt='%1.4f')

"""
\rho = 0.5 a, ordered
"""
np.savetxt('05_T_pRay_order.txt', abs(SE0_order05.Transmittance)**2, newline='`,',fmt='%1.4f')
np.savetxt('05_R_pRay_order.txt', abs(SE0_order05.Reflection)**2, newline='`,',fmt='%1.6f')

np.savetxt('05_T_fRay_order.txt', SE0_order05.fullTransmittance, newline='`,',fmt='%1.4f')
np.savetxt('05_R_fRay_order.txt', SE0_order05.fullReflection, newline='`,',fmt='%1.6f')

np.savetxt('05_T_sRaman_order.txt', SE1_order05.fullTransmittance, newline='`,',fmt='%1.4f')
np.savetxt('05_R_sRaman_order.txt', SE1_order05.fullReflection, newline='`,',fmt='%1.6f')

"""
\rho = 0.5 a, disordered
"""


np.savetxt('05_T_pRay_disorder.txt', abs(SE0_disorder05.Transmittance)**2, newline='`,',fmt='%1.4f')
np.savetxt('05_R_pRay_disorder.txt', abs(SE0_disorder05.Reflection)**2, newline='`,',fmt='%1.6f')

np.savetxt('05_T_fRay_disorder.txt', SE0_disorder05.fullTransmittance, newline='`,',fmt='%1.4f')
np.savetxt('05_R_fRay_disorder.txt', SE0_disorder05.fullReflection, newline='`,',fmt='%1.6f')

np.savetxt('05_T_sRaman_disorder.txt', SE1_disorder05.fullTransmittance, newline='`,',fmt='%1.4f')
np.savetxt('05_R_sRaman_disorder.txt', SE1_disorder05.fullReflection, newline='`,',fmt='%1.6f')

"""
\rho = 1.0 a  ordered
"""

np.savetxt('10_T_pRay_order.txt', abs(SE0_order10.Transmittance)**2, newline='`,',fmt='%1.4f')
np.savetxt('10_R_pRay_order.txt', abs(SE0_order10.Reflection)**2, newline='`,',fmt='%1.6f')

np.savetxt('10_T_fRay_order.txt', SE0_order10.fullTransmittance, newline='`,',fmt='%1.4f')
np.savetxt('10_R_fRay_order.txt', SE0_order10.fullReflection, newline='`,',fmt='%1.6f')

np.savetxt('10_T_sRaman_order.txt', SE1_order10.fullTransmittance, newline='`,',fmt='%1.4f')
np.savetxt('10_R_sRaman_order.txt', SE1_order10.fullReflection, newline='`,',fmt='%1.8f')

print('done')

"""
\rho = 1.0 a, disordered
"""

np.savetxt('10_T_pRay_disorder.txt', abs(SE0_disorder10.Transmittance)**2, newline='`,',fmt='%1.4f')
np.savetxt('10_R_pRay_disorder.txt', abs(SE0_disorder10.Reflection)**2, newline='`,',fmt='%1.6f')

np.savetxt('10_T_fRay_disorder.txt', SE0_disorder10.fullTransmittance, newline='`,',fmt='%1.4f')
np.savetxt('10_R_fRay_disorder.txt', SE0_disorder10.fullReflection, newline='`,',fmt='%1.6f')

np.savetxt('10_T_sRaman_disorder.txt', SE1_disorder10.fullTransmittance, newline='`,',fmt='%1.4f')
np.savetxt('10_R_sRaman_disorder.txt', SE1_disorder10.fullReflection, newline='`,',fmt='%1.6f')





"""
__________________________________________________
Bragg effect
__________________________________________________

"""

Nof_dots_b = 10
dlambda = np.linspace(0.5, 1.5, Nof_dots_b)

args = args_def.copy()
Tmin = np.zeros_like(dlambda)
Rmax = np.zeros_like(dlambda)
for k in range(Nof_dots_b):
    args['l0'] = dlambda[k]
    chi = ods.ensemble(**args)
    chi.generate_ensemble()

    Tmin[k] = np.min(chi.fullTransmittance)
    Rmax[k] = np.max(chi.fullReflection)

args = args_def.copy()
args['d'] = 1.5
Tminc = np.zeros_like(dlambda)
Rmaxc = np.zeros_like(dlambda)
for k in range(Nof_dots_b):
    args['l0'] = dlambda[k]
    chi = ods.ensemble(**args)
    chi.generate_ensemble()

    Tminc[k] = np.min(chi.fullTransmittance)
    Rmaxc[k] = np.max(chi.fullReflection)



plt.plot(dlambda, Tmin, 'b-', label = r'$\rho =  a$')
plt.plot(dlambda, Tminc, 'r--', label = r'$\rho = 0.5 a$')
plt.xlabel('Distance between atoms,$\lambda/2$')
plt.ylabel('Transmission')
plt.legend()
plt.show()

plt.plot(dlambda, Rmax, 'b-', label = r'$\rho =  a$')
plt.plot(dlambda, Rmaxc, 'r--', label = r'$\rho = 0.5 a$')
plt.xlabel('Distance between atoms, $\lambda/2$')
plt.ylabel('Reflection')
plt.legend()
plt.show()



"""
__________________________________________________
Loss (number of atoms) (for correlated chain only!)
__________________________________________________

"""


Nof_dots_l = 7
args = args_def.copy()
Tnum = np.zeros(Nof_dots_l-1)
Rnum = np.zeros(Nof_dots_l-1)
for k in range(1,Nof_dots_l):

    args['nat'] = k
    args['nb'] = k-1
    chi = ods.ensemble(**args)
    chi.generate_ensemble()

    Tnum[k-1] = np.min(chi.fullTransmittance)
    Rnum[k-1] = np.max(chi.fullReflection)


args = args_def.copy()
args['d'] = 1.5
Tnumc = np.zeros(Nof_dots_l-1)
Rnumc = np.zeros(Nof_dots_l-1)
for k in range(1,Nof_dots_l):

    args['nat'] = k
    args['nb'] = k-1
    chi = ods.ensemble(**args)
    chi.generate_ensemble()

    Tnumc[k-1] = np.min(chi.fullTransmittance)
    Rnumc[k-1] = np.max(chi.fullReflection)

"""
Plotting
"""
plt.plot(range(1,Nof_dots_l),Tnum, 'b-', label = r'$\rho =  a$')
plt.plot(range(1,Nof_dots_l),Tnumc, 'r--', label = r'$\rho = 0.5 a$')
plt.xlabel('Number of atoms')
plt.ylabel('Transmission')
plt.legend()
plt.show()

plt.plot(range(1,Nof_dots_l),Rnum, 'b-', label = r'$\rho =  a$')
plt.plot(range(1,Nof_dots_l),Rnumc, 'r--', label = r'$\rho = 0.5 a$')
plt.xlabel('Number of atoms')
plt.ylabel('Reflection')
plt.legend()
plt.show()

