import numpy as np
import matplotlib.pyplot as plt

n = np.linspace(1.1, 2.1, 100)
a = 2.405 / (2*np.pi *np.sqrt(n**2 - 1))

plt.plot(n,a)
plt.xlabel('refraction index')
plt.ylabel('radius,' + '$\lambda$')

plt.plot(1.45, 200/780, 'ro')

plt.plot(1.45, 250/780, 'bo')

plt.plot(1.25, 200/780, 'bo')

plt.plot(1.25, 300/780, 'bo')

plt.plot(1.2, 400/780, 'bo')
plt.plot(1.6, 200/780, 'bo')
plt.show()