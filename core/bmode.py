# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 16:45:02 2016

@author: ViacheslavP
"""
import numpy as np
from scipy.special import jn, kn


class exact_mode(object):

    """
       Initially, the dipole gauge is used all over the calculation, so, the critical step is to define k (or beta) in
    some small region of root of the equation $\omega^2_x = \omega^2$, such that
    $\omega^2(k) = \omega^2 + v_f v_g (k^2 - x^2)$. It's done in the _second_initialization subroutine.
    The second step is routine: defining field functions (spatial dependence) for particular k.

    The syntax:

    mode = bmode.exact_mode(1, 1.45, 200/780)
    mode.generate_mode()
    ex = mode.E(atomic positions)

    ...

    mode.generate_sub_mode()
    sub_ex = mode.E(atomic positions)

    ...

    """
    
    def __init__(self, omega, n, a):
        """
        :param float omega: the frequency of fundamental mode
        :param float n: the refraction index
        :param float a: the waveguide radius, in $\lambdabar$ units
        """
        self._n = n
        self._s = 0.
        self._omega = omega
        self.k = omega
        self._norm = 1.
        self.vg = 0
        self._a = a

        v = omega * a * np.sqrt(n ** 2 - 1)
        if v > 2.405:
            print("ERROR: Multi mode regime. V = ", v)
            raise ValueError

        self._second_initialization_()
        self.generate_mode()

    def _second_initialization_(self):

        """
        Due to the fact that characteristic equation contains various bessel
        functions, I am using symbolical evaluation to compute derivative. The alter way is to use
        high order approximation of derivatives (which is not seems reasonable!).
        """

        import sympy as sp
        from scipy.optimize import newton_krylov as krlv

        x, _k = sp.symbols('x,_k')  # x stands for propagation const., _k stands for omega
        n = self._n
        a = self._a

        ha = sp.sqrt(n * n * _k * _k - x * x) * a
        qa = sp.sqrt(x * x - _k * _k) * a
        kd = (-sp.besselk(0, qa) / 2 - sp.besselk(2, qa) / 2) * qa / sp.besselk(1, qa)
        jd = (sp.besselj(0, ha) / 2 - sp.besselj(2, ha) / 2) * ha / sp.besselj(1, ha)

        ceq = ((jd * (n * qa / ha) ** 2 + kd) * (jd * (qa / ha) ** 2 + kd)) - (
        ((n * n - 1) * x * _k * a * a / ha / ha) ** 2)
        fo = sp.diff(ceq, _k)
        fk = sp.diff(ceq, x)

        bessel = {'besselj': jn, 'besselk': kn, 'sqrt': np.sqrt}
        libraries = [bessel, "numpy"]



        foo_eq = sp.lambdify([x, _k], sp.re(ceq), modules=libraries)
        eig_eq = lambda x: foo_eq(x, self._omega)

        """
        _______________________________________
        Subroutine output:
        1. Propagation const.
        2. Group velocity
        3. Phase velocity
        4. $\omega^2_k$ in domain of current propagation const.
        _______________________________________
        """

        self.k = krlv(eig_eq, self._omega * (self._n + 1) / 2)
        self.vg = sp.lambdify([x, _k], -fk / fo, modules=libraries)(self.k, self._omega)
        self.vp = self._omega / self.k
        self.sqr_omega = lambda kl: self._omega**2 + self.vg*self.vp * (kl**2 - self.k**2)


    def _expressions_(self, k):
        """
        :param k: the propagation constant (momentum, due to translation sym)

        The method sets the output of field components for particular k

        Mode components as it presented in Balykin's paper - more convenient to numerical simulations.
        Cylindrical components are exact as in fundamental_mode.py
        Decomposition components are multiplied by sqrt(2) (discuss!! It was my fault)
        The polarization components of the mode are:
        E- =  E e- + dE exp(-2i phi) e+ + Ez exp(-i phi) ez

        To get the opposite polarisation:
        'r$ E_{-\sigma}^{\mu} = (E_{\sigma}^{-\mu})^{\asr} $'

        It is normalized to be $ \int \epsilon(r) |E(r)| ^ 2 dx dy = \lambdabar ^ 2 (x,y in cm),
                            or  \int \epsilon(r) |E(r)| ^ 2 dx dy = 1 (x,y in \lambdabar) $
        """
        from scipy.integrate import quad

        a = self._a
        omega = np.sqrt(self.sqr_omega(k))

        n = self._n
        ha = a * np.sqrt(n*n*omega*omega - k*k)
        qa = a * np.sqrt(k**2- omega**2)
        s = (1 / qa / qa + 1 / ha / ha) / \
            ((-kn(0, qa) - kn(2, qa)) / 2 / qa / kn(1, qa) + (jn(0, ha) - jn(2, ha)) / 2 / ha / jn(1, ha))

        efin = lambda r: -1 * self.k / 2 / ha * a * (1 - s) * jn(0, ha * r / a) * np.sqrt(2) / jn(1, ha)
        efout = lambda r: -1 * self.k / 2 / qa * a * (1 - s) * kn(0, qa * r / a) * np.sqrt(2) / kn(1, qa)

        E = lambda r: np.piecewise(r, [r >= a, r < a], [efout, efin])

        dfin = lambda r: -1 * self.k / 2 / ha * a * (1 + s) * jn(2, ha * r / a) * np.sqrt(2) / jn(1, ha)
        dfout = lambda r: 1 * self.k / 2 / qa * a * (1 + s) * kn(2, qa * r / a) * np.sqrt(2) / kn(1, qa)

        dE = lambda r: np.piecewise(r, [r >= a, r < a], [dfout, dfin])

        zfin = lambda r: jn(1, ha * r / a) / jn(1, ha)
        zfout = lambda r: kn(1, qa * r / a) / kn(1, qa)

        Ez = lambda r: np.piecewise(r, [r >= a, r < a], [zfout, zfin])

        _fooint = lambda r: r * (abs(E(r)) ** 2 + abs(dE(r)) ** 2 + abs(Ez(r)) ** 2)
        _norm = np.sqrt((quad(_fooint, a, np.inf)[0] + n * n * quad(_fooint, 0, a)[0]) * 2 * np.pi)


        self.E = lambda r: 1j * E(r) / _norm
        self.dE = lambda r: 1j * dE(r) / _norm
        self.Ez = lambda r: Ez(r) / _norm
        self.Ephi = lambda r: 1j / np.sqrt(2) * (self.E(r) + self.dE(r))
        self.Er = lambda r: 1 / np.sqrt(2) * (self.E(r) - self.dE(r))

        # H-field

        eps = n * n

        hzi = lambda r: self.k * s * jn(1, ha * r / a) / jn(1, ha) / _norm
        hzo = lambda r: self.k * s * kn(1, ha * r / a) / kn(1, ha) / _norm

        self.Hz = lambda r: 1j * np.piecewise(r, [r >= a, r < a], [hzo, hzi])

        hfi = lambda r: ((eps - s * (self.k) ** 2) * jn(0, ha * r / a) - (eps + s * (self.k) ** 2) * jn(2,
                                                                                                        ha * r / a)) * a / 2 / ha / jn(
            1, ha) / _norm
        hfo = lambda r: ((1 - s * (self.k) ** 2) * kn(0, qa * r / a) + (1 + s * (self.k) ** 2) * kn(2,
                                                                                                    qa * r / a)) * a / 2 / qa / kn(
            1, qa) / _norm

        self.Hphi = lambda r: 1j * np.piecewise(r, [r >= a, r < a], [hfo, hfi])

        hri = lambda r: ((eps - s * (self.k) ** 2) * jn(0, ha * r / a) + (eps + s * (self.k) ** 2) * jn(2,
                                                                                                        ha * r / a)) * a / 2 / ha / jn(
            1, ha) / _norm
        hro = lambda r: ((1 - s * (self.k) ** 2) * kn(0, qa * r / a) - (1 + s * (self.k) ** 2) * kn(2,
                                                                                                    qa * r / a)) * a / 2 / qa / kn(
            1, qa) / _norm

        self.Hr = lambda r: 1 * np.piecewise(r, [r >= a, r < a], [hro, hri])

    def generate_mode(self):

        """
        Switch class to fundamental mode ($\omega^2_k = \omega^2$)
        """

        self._expressions_(self.k)

    def generate_sub_mode(self, rho = 1.5*200/780*2*np.pi):
        from scipy.integrate import quad as quad
        from matplotlib import pyplot as plt
        self.generate_mode()
        norm = self._norm

        def fD(k,q):
            a = self._a
            omega = np.sqrt(self.sqr_omega(k))
            n = self._n

            ha = a * np.sqrt(n * n * omega * omega - k * k)
            qa = a * np.sqrt(k * k - omega * omega)
            s = (1 / qa / qa + 1 / ha / ha) / \
                ((-kn(0, qa) - kn(2, qa)) / 2 / qa / kn(1, qa) + (jn(0, ha) - jn(2, ha)) / 2 / ha / jn(1, ha))



            return -1*k*(1-s)/norm*2*a*np.pi*(n*n*(jn(0,q*a)*jn(1,ha)*ha - jn(0,ha)*jn(1,q*a)*q*a) /
                                              (ha*jn(1,ha)*(n*n*omega*omega - k*k - q*q)) -
                                              (jn(0,q*a)*kn(1,qa)*qa - kn(0,qa)*jn(1,q*a)*q*a) /
                                              (qa*kn(1,qa)*(omega*omega - k*k - q*q)))

        g = lambda k,q: q *jn(0,q*rho) / 2 / np.pi * fD(k,q)
        A = np.inf
        def denominator(k):
            num = lambda x: g(k,x) / (self._omega**2 - k**2 - x**2)
            den = lambda x: g(k,x)
            return quad(num, 0, A, epsabs = 1e-9)[0] / quad(den, 0, A, epsabs=1e-9)[0]
        print(fD(0.01,0.99),"HOLA!")
        xn  = np.linspace(0.99, 1.095, 110)
        foo = lambda x: (self._omega**2-x**2-1/denominator(x))
        yn = np.array([foo(x) for x in xn])
        plt.plot(xn,yn, 'r')
        plt.show()
        #self._expressions_(self.sub_k)
    def plot_HankelImage(self):
        from scipy.integrate import quad
        from matplotlib import pyplot as plt
        he = lambda x: 1 + (1.45**2-1)* np.piecewise(x, [x < 1, x > 1], [1, 0])

        q = np.linspace(0,5,100)
        e = np.zeros_like(q)
        de = np.zeros_like(q)
        ez = np.zeros_like(q)

        for i in range(len(q)):
            e[i] =  quad(lambda x: -np.imag(self.E(x)*x*jn(0,x*q[i]))*he(x)*2*np.pi, 0, np.inf)[0]
            de[i] = quad(lambda x: np.imag(self.dE(x)*x*jn(2,x*q[i]))*he(x)*2*np.pi, 0, np.inf)[0]
            ez[i] = quad(lambda x: self.Ez(x)*x*jn(1,x*q[i])*he(x)*2*np.pi, 0, np.inf)[0]

        plt.plot(q,e, label=r'$D_{\perp}(q)$')
        plt.plot(q,de, label=r'$D^{\prime}(q)$')
        plt.plot(q,ez, label=r'$D^{\prime \prime}(q)$')
        plt.legend(fontsize=18)
        plt.show()








if __name__ == '__main__':
    args = {'omega':1,
            'n':1.45,
            'a': 2*np.pi*200/850
            }
    a = 2*np.pi*200/850
    m = exact_mode(**args)
    m.generate_mode()
    print(m.E(2.))
    #m.plot_HankelImage()
"""
    from matplotlib import pyplot as plt

    he = lambda x: 1+1.1*np.piecewise(x, [x<1,x>1], [1,0])
    x = np.linspace(0,2,100)
    y_0 = he(x)*abs(m.E(x*a))
    y_1 = he(x)*abs(m.Ez(x*a))
    y_2 = he(x)*np.imag(m.dE(x*a))

    plt.plot(x, y_0, label=r'$D_{\perp}(\rho)$',lw=2.)
    plt.plot(x, y_1, label=r'$D^{\prime \prime}(\rho)$',lw=2.)
    plt.plot(x, y_2, label=r'$D^{\prime}(\rho)$',lw=2.)
    plt.xlabel(r'$\rho/a$',fontsize=20)
    plt.legend(fontsize=20)
    plt.show()
"""
