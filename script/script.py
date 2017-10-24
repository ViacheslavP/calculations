from bmode import exact_mode
from matplotlib import pyplot as plt
import numpy as np

def gauss(x,s=3):
    return 1/np.sqrt(np.pi*s*s) * np.exp(-x*x/s/s)

args = {'k':1,
        'n':1.45,
        'a': 2*np.pi*200/780
       }
m = exact_mode(**args)
a =  2*np.pi*200/780
x = np.linspace(0,5, 100)
y = abs(m.E(x))
yg = gauss(x, 3.2)

plt.plot(x,y)
plt.plot(x,yg)
plt.plot([a*1.5], 0.1, 'o')
plt.plot([a*2], 0.065, 'o')
plt.show()