import sys, os

sys.path.insert(0, 'core/')
import one_D_scattering as ods
import numpy as np

from scipy.integrate import cumtrapz as ct
from wave_pack import convolution, delay
from wave_pack import inverse_pulse as pulse
from wave_pack import efficiency

ods.PAIRS = True

# Control mode Setup
ods.RABI = 0
ods.DC = -4  # 6#7

# Pulse setup
pdGamma = 0.05  # 26.7#26.9#2*np.pi
pdShift = +4

# Number of atoms
noa = 1100

# Calculation Scheme Setup
ods.RABI_HYP = False
ods.SINGLE_RAMAN = True
ods.RADIATION_MODES_MODEL = 1
ods.VACUUM_DECAY = 1
ods.DDI = 1
ods.EDGE_STABILITY = False
raz = 9600
freq = np.linspace(-1600.5, 1600.5, raz)
CSVPATH = 'data/m_pics_15.03.2020/'
if not os.path.exists(CSVPATH):
    os.makedirs(CSVPATH)
# First figure

args = {
    'nat': noa,  # number of atoms
    'nb': 0,  # number of neighbours in raman chanel (for L-atom only)
    's': 'chain',  # Stands for atom positioning : chain, nocorrchain and doublechain
    'dist': 0.,  # sigma for displacement (choose 'chain' for gauss displacement., \lambda/2 units)
    'd': 1.5,  # distance from fiber
    'l0': 2.00 / 2,  # mean distance between atoms (in lambda_m /2 units)
    'deltaP': freq,  # array of freq
    'typ': 'LM',  # L or V for Lambda and V atom resp.
    'ff': 0.3,  # filling factor (for ff_chain only)
    'numexc': 0
}

from scipy.optimize import curve_fit


def extract_time(t, et):
    def _func(x, g, A):
        return A * np.exp(-1 * x * g)  # np.A*np.exp(-x*g)

    lent = len(t)
    lens = np.int(lent // 1.8)
    t = t[lens:lent]
    et = et[lens:lent]

    # plt.plot(t, et)
    # plt.plot(t, et - np.amax(et)*_func(t,1,1))
    # plt.show()

    popt, pcov = curve_fit(_func, t, et / np.amax(et), bounds=(0, 3), method='trf')

    return popt[0]


def smooth(t, et):
    def _func(t, g1, g2, g3, A1, A2, A3, rabi):
        return A1 * np.exp(-1 * g1 * t) + A2 * np.exp(-1 * g2 * t) + A3 * np.exp(-1 * g3 * t)

    lent = len(t)
    lens = lent // 2

    t = t[lens:lent];
    et = np.real(et[lens:lent])

    popt, pcov = curve_fit(_func, t, et / np.amax(et), method='trf')

    et = _func(t, *popt) / (popt[3] + popt[4] + popt[5])

    return t, et / et.max()

def isiinthemiddle(i, noa, notright=False):
    if i < noa:
        ni = i
    else:
        ni = (i - noa) % (2 * (noa - 1)) // 2

    n1 = np.int(0.5 * (1 - ods.ACTIVE) * noa)
    n2 = np.int(0.5 * (1 + ods.ACTIVE) * noa)
    if ni >= n1 and ni < n2:
        iisright=True
    else:
        iisright=notright

    return iisright

def isiintheside(i, noa, notright=False):
    if i < noa:
        ni = i
    else:
        ni = (i - noa) % (2 * (noa - 1)) // 2

    n1 = noa * ods.ACTIVE
    if ni <= n1:
        iisright = True
    else:
        iisright = notright

    return iisright

ods.PAIRS = False
ods.PHASE = np.pi
ods.RANDOM_PHASE = False
ods.RADIATION_MODES_MODEL = True

#First line - dicke state decay
if True:
    ods.ACTIVE = 100/1100
    ods.RADIATION_MODES_MODEL = 1
    ods.VACUUM_DECAY = 1
    ods.PAIRS = False
    ods.RANDOM_PHASE = False
    ods.PHASE = np.pi
    ods.EDGE_STABILIY = False
    args['s'] = 'chain'
    args['nat'] = np.int( noa * ods.ACTIVE )
    args['d'] = 1.5
    dicke = ods.ensemble(**args)
    dicke.generate_ensemble()

    dAD = np.zeros_like(dicke.AtomicDecay)

    for i in range(len(dicke.AtomicDecay[:, 0])):
        t, dAD[i, :] = convolution(freq, dicke.AtomicDecay[i, :], np.ones_like(freq))

    dickeDecay = np.dot(np.conj(np.transpose(dAD)), dAD).diagonal()

#Second line - resonator|projected|initial state decay
if True:
    ods.RADIATION_MODES_MODEL = 1
    ods.VACUUM_DECAY = 1
    ods.PAIRS = False
    ods.RANDOM_PHASE = False
    ods.PHASE = np.pi
    ods.EDGE_STABILIY = False
    args['nat'] = noa
    args['s'] = 'custom-resonator'
    args['d'] = 1.5
    ods.LMDA = 0 * 1.
    resonator = ods.ensemble(**args)
    resonator.generate_ensemble()

    rAD = np.zeros_like(resonator.AtomicDecay)

    for i in range(len(resonator.AtomicDecay[:, 0])):

        if isiinthemiddle(i, noa, True):
            _, rAD[i, :] = convolution(freq, resonator.AtomicDecay[i, :], np.ones_like(freq))

    resDecay = np.dot(np.conj(np.transpose(rAD)), rAD).diagonal()

    rAD = np.zeros_like(resonator.AtomicDecay)

    for i in range(len(resonator.AtomicDecay[:, 0])):

        if isiinthemiddle(i, noa, False):
            _, rAD[i, :] = convolution(freq, resonator.AtomicDecay[i, :], np.ones_like(freq))

    projresDecay = np.dot(np.conj(np.transpose(rAD)), rAD).diagonal()

    ddRight_in = np.zeros_like(resonator.AtomicDecay[:, 0])
    ddRight_full = np.zeros_like(resonator.AtomicDecay[:, 0])

    for i in range(noa):
        if isiinthemiddle(i, noa, False):
            ddRight_in[i] = (-1) ** i

        ddRight_full[i] = (-1) ** i

    t, blochIn = convolution(freq, np.dot(ddRight_in, resonator.AtomicDecay), np.ones_like(freq))
    inDecay = abs(blochIn) ** 2

    t, blochFull = convolution(freq, np.dot(ddRight_full, resonator.AtomicDecay), np.ones_like(freq))
    fullDecay = abs(blochFull) ** 2

    dcay = np.exp(-t)
    ts, _ddecay = t, np.real(dickeDecay)
    lines = np.empty((len(ts), 8), dtype=np.float32)


    def toMathematica(fname, *argv):
        toCsv = np.column_stack(argv)
        np.savetxt(CSVPATH + fname + '.csv', toCsv, delimiter=',', fmt='%1.8f')

    lines[:, 0], lines[:, 1] = t, np.real(dickeDecay)
    lines[:, 2] = dickeDecay
    lines[:, 3] = resDecay
    lines[:, 4] = projresDecay
    lines[:, 5] = dcay
    lines[:, 6] = inDecay
    lines[:, 7] = fullDecay

    toMathematica('rg_res', lines)

    ts, _ddecay = smooth(t, np.real(dickeDecay))
    lines = np.empty((len(ts), 8), dtype=np.float32)

    lines[:, 0], lines[:, 1] = smooth(t, np.real(dickeDecay))
    _, lines[:, 2] = smooth(t, dickeDecay)
    _, lines[:, 3] = smooth(t, resDecay)
    _, lines[:, 4] = smooth(t, projresDecay)
    _, lines[:, 5] = smooth(t, dcay)
    _, lines[:, 6] = smooth(t, inDecay)
    _, lines[:, 7] = smooth(t, fullDecay)
    # _, lines[:, 8] = smooth(t, bigSidemDecay/bigSidemDecay.max())

    toMathematica('sm_res', lines)

#side-mirror
if True:
    ods.RADIATION_MODES_MODEL = 1
    ods.VACUUM_DECAY = 1
    ods.PAIRS = False
    ods.RANDOM_PHASE = False
    ods.PHASE = np.pi
    ods.EDGE_STABILIY = False
    args['nat'] = noa
    args['s'] = 'custom-side-mirror'
    args['d'] = 1.5
    ods.LMDA = 0 * 1.
    resonator = ods.ensemble(**args)
    resonator.generate_ensemble()

    rAD = np.zeros_like(resonator.AtomicDecay)

    for i in range(len(resonator.AtomicDecay[:, 0])):

        if isiintheside(i, noa, True):
            _, rAD[i, :] = convolution(freq, resonator.AtomicDecay[i, :], np.ones_like(freq))

    smDecay = np.dot(np.conj(np.transpose(rAD)), rAD).diagonal()

    rAD = np.zeros_like(resonator.AtomicDecay)

    for i in range(len(resonator.AtomicDecay[:, 0])):

        if isiintheside(i, noa, False):
            _, rAD[i, :] = convolution(freq, resonator.AtomicDecay[i, :], np.ones_like(freq))

    projresDecay = np.dot(np.conj(np.transpose(rAD)), rAD).diagonal()

    ddRight_in = np.zeros_like(resonator.AtomicDecay[:, 0])
    ddRight_full = np.zeros_like(resonator.AtomicDecay[:, 0])

    for i in range(noa):
        if isiintheside(i, noa, False):
            ddRight_in[i] = (-1) ** i

        ddRight_full[i] = (-1) ** i

    t, blochIn = convolution(freq, np.dot(ddRight_in, resonator.AtomicDecay), np.ones_like(freq))
    inDecay = abs(blochIn) ** 2

    t, blochFull = convolution(freq, np.dot(ddRight_full, resonator.AtomicDecay), np.ones_like(freq))
    fullDecay = abs(blochFull) ** 2

    dcay = np.exp(-t)
    ts, _ddecay = t, np.real(dickeDecay)
    lines = np.empty((len(ts), 8), dtype=np.float32)


    def toMathematica(fname, *argv):
        toCsv = np.column_stack(argv)
        np.savetxt(CSVPATH + fname + '.csv', toCsv, delimiter=',', fmt='%1.8f')


    lines[:, 0], lines[:, 1] = t, np.real(dickeDecay)
    lines[:, 2] = dickeDecay
    lines[:, 3] = resDecay
    lines[:, 4] = projresDecay
    lines[:, 5] = dcay
    lines[:, 6] = inDecay
    lines[:, 7] = fullDecay

    toMathematica('rg_sdm', lines)

    ts, _ddecay = smooth(t, np.real(dickeDecay))
    lines = np.empty((len(ts), 8), dtype=np.float32)

    lines[:, 0], lines[:, 1] = smooth(t, np.real(dickeDecay))
    _, lines[:, 2] = smooth(t, dickeDecay)
    _, lines[:, 3] = smooth(t, resDecay)
    _, lines[:, 4] = smooth(t, projresDecay)
    _, lines[:, 5] = smooth(t, dcay)
    _, lines[:, 6] = smooth(t, inDecay)
    _, lines[:, 7] = smooth(t, fullDecay)
    # _, lines[:, 8] = smooth(t, bigSidemDecay/bigSidemDecay.max())

    toMathematica('sm_sdm', lines)


try:
    from matplotlib import pyplot as plt


    def smtplot(x, y, *args, **kwargs):
        x1, y1 = smooth(x, y / y.max())
        plt.plot(x1, y1, *args, **kwargs)


    def decomposeplot(x, y, *args, **kwargs):
        def _func(t, g1, g2, A1, A2, rabi, A3):
            return A1 * np.exp(-1 * g1 * t) + A2 * np.exp(-1 * g2 * t)

        lent = len(x)
        lens = lent // 2

        xt = x[lens:lent];
        yt = np.real(y[lens:lent])

        popt, pcov = curve_fit(_func, xt, yt / np.amax(yt), method='trf')

        x1, y1 = x, np.exp(-1 * popt[0] * x)
        x2, y2 = x, np.exp(-1 * popt[1] * x)
        plt.plot(x1, y1, *args, **kwargs)
        plt.plot(x2, y2, *args, **kwargs)


    def normalplot(t, et, *args, **kwargs):
        def _func(t, g1, A1):
            return A1 * np.exp(-1 * g1 * t)

        lent = len(t)
        lena = lent // 2
        lenb = lent // 2 + lent // 20

        t1 = t[lena:lenb];
        et1 = np.real(et[lena:lenb])

        popt, pcov = curve_fit(_func, t1, et1, method='trf')

        plt.plot(t, et / _func(0.2, *popt), *args, **kwargs)

    plt.rcParams.update({'font.size': 16})
    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.grid(True, axis='y', which='both')
    plt.xlim([0,2])
    plt.ylim([0.1, 1.])
    plt.yscale('log')
    dcay = np.exp(-t)
    smtplot(t, (dickeDecay/dickeDecay.max()), 'r')
    smtplot(t, resDecay/resDecay.max(), 'g')
    smtplot(t, projresDecay/projresDecay.max(), 'b-')
    smtplot(t, inDecay/inDecay.max(), 'r--')
    smtplot(t, fullDecay/fullDecay.max(), 'b-.')
    #plt.plot(t, extresDecay/resDecay.max()/0.8, 'g')
    #plt.plot(t, extshiftresDecay/shiftresDecay.max()/0.8/0.88, 'g--')
    #smtplot(t, dresDecay/dresDecay.max(), 'm-.')
    #smtplot(t, rresDecay/rresDecay.max(), 'c--')
    #smtplot(t, dcay, 'k--')
    #plt.plot(t, sidemDecay/sidemDecay.max(), 'b')
    #smtplot(t, bigResDecay, 'r--')
    #smtplot(t, bigSidemDecay/bigSidemDecay.max(), 'r-.')
    plt.xlabel(r'$\gamma t$')
    plt.ylabel(r'$p(t)$')
    plt.savefig(CSVPATH + 'ms_stableedg.svg')
    plt.savefig('plot.svg')

except:
    print('There will be no figure')

dcay = np.exp(-t)
ts, _ddecay = smooth(t, np.real(dickeDecay))
lines = np.empty((len(ts), 8), dtype=np.float32)


def toMathematica(fname, *argv):
    toCsv = np.column_stack(argv)
    np.savetxt(CSVPATH + fname + '.csv', toCsv, delimiter=',', fmt='%1.8f')



lines[:, 0], lines[:, 1] = smooth(t, np.real(dickeDecay))
_, lines[:, 2] = smooth(t, dickeDecay)
_, lines[:, 3] = smooth(t, resDecay)
_, lines[:, 4] = smooth(t, projresDecay)
_, lines[:, 5] = smooth(t, dcay)
_, lines[:, 6] = smooth(t, inDecay)
_, lines[:, 7] = smooth(t, fullDecay)
# _, lines[:, 8] = smooth(t, bigSidemDecay/bigSidemDecay.max())

toMathematica('comparison', lines)