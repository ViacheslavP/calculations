import numpy as np
import one_D_scattering as ods
from matplotlib import pyplot as plt






"""
___________________________________________________

--------------------Decay--------------------------
___________________________________________________
"""
from bmode import exact_mode

a = 2 * np.pi / 780
ndots = 41
c = 1
kv = 1
d00 = 0.5

arg0 = {'omega': 1,
        'n': 1.452,
        'a': 200*a
         }

arg1 = {'omega': 1,
        'n': 1.45,
        'a': 250*a
         }

arg2 = {'omega': 1,
        'n': 1.25,
        'a': 200*a
         }

arg3 = {'omega': 1,
        'n': 1.25,
        'a': 300*a
         }

arg4 = {'omega': 1,
        'n': 1.2,
        'a': 400*a
         }

arg5 = {'omega': 1,
        'n': 1.6,
        'a': 200*a
         }

x = np.linspace(1, 5, ndots)
np.savetxt('verification_multi/x.txt', x, newline='`,',fmt='%1.7f')

argar = [arg0, arg1, arg2, arg3, arg4, arg5]
m = []
i = 0
for arg in argar:

    gd = np.zeros(ndots)
    m = exact_mode(**arg)
    vg = m.vg
    ep = m.E(x*arg['a'])
    em = m.dE(x*arg['a'])
    ez = m.Ez(x*arg['a'])
    for i in range(ndots):
        gd[i] = 1 + 8 * d00 * d00 * np.pi *  (
        (1 / vg ) * (abs(em[i]) ** 2 + abs(ep[i]) ** 2 + \
            abs(ez[i]) ** 2) - 1 / c * abs(ep[i]) ** 2 )

    np.savetxt('verification_multi/arg_(%f03)_(%g).txt' % (arg['a']*780/2/np.pi, arg['n']), np.real(gd), newline='`,', fmt='%1.7f')
    i=i+1



