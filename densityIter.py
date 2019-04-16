import numpy as np
import scipy.stats as stats

sigma = 2.
L = 10
dq = 2*np.pi
rho = lambda z: (np.cos(np.pi * z / L)*np.cos(dq * z))**2
eps = 0.001

np.random.seed(seed=1)
def iterate_density():
    x0 = 0
    x1 = L
    while True:
        x1 = np.random.normal(x0, sigma)
        if x1 >= L/2 or x1 <= -L/2:
            continue

        acceptance = (rho(x1)/rho(x0))

        uniform_number = np.random.uniform(0,1)

        if acceptance < uniform_number:
            continue
        else:
            x0 = x1
            yield x1

def rhoMCMC(num):
    zns = np.empty(num, dtype=np.float)
    iter = 0
    for zn in iterate_density():
        zns[iter] = zn
        iter+=1
        if iter == num:
            break
    return zns

if __name__ == '__main__':
    N = 500
    import matplotlib.pyplot as plt
    plt.hist(rhoMCMC(N), 100, histtype='step', lw=2)

    plt.plot(np.linspace(-L/2, L/2, 1000), rho(np.linspace(-L/2, L/2, 1000)))
    plt.show()