# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 16:45:02 2016

@author: ViacheslavP
"""
import numpy as np
from scipy.special import jn,kn
import os


class exact_mode(object):
    
    def __init__(self,k,n,a):
        self._n = n
        self._s = 0.
        self._k = k
        self._beta = k
        self._norm = 1.
        self.vg = 0
        self._a = a        
        
        self.E = lambda x: x
        self.dE = lambda x: x       
        self.Ez = lambda x: x
        
        self.Er = lambda x: x
        self.Ephi = lambda x: x 
        
        self.generate_mode()
        
        
    def generate_mode(self):
        """
        1.Due to the fact that characteristic equation contains various bessel
        functions, I am using symbolical evaluation to compute derivative. The alter way is to use 
        high order approximation of derivatives (which is not seems reasonable!).
        ceq(x,_k) = 0 - the spectral problem equation
        x = beta
        _k = omega/c
        2.In multimode regime some sort of bias could happen (clearly: It will calculate all wrong!). 

        """
        
        import sympy as sp
        from scipy.optimize import excitingmixing as brd
        from scipy.optimize import newton_krylov as krlv
        from scipy.integrate import quad
        from scipy.optimize import curve_fit as cf
        
        
        """
        _______________________________________
        Characteristic equation
        _______________________________________
        """
        x,_k = sp.symbols('x,_k')
        n = self._n
        a = self._a
        
        ha = sp.sqrt(n*n*_k*_k - x*x)*a
        qa = sp.sqrt(x*x - _k*_k)*a
        kd = (-sp.besselk(0, qa)/2 - sp.besselk(2, qa)/2)*qa/sp.besselk(1,qa)
        jd = (sp.besselj(0, ha)/2 - sp.besselj(2, ha)/2)*ha/sp.besselj(1,ha)
        
        ceq = ((jd*(n*qa/ha)**2+kd)*(jd*(qa/ha)**2+kd))  - (((n*n-1)*x*_k*a*a/ha/ha)**2) 
        fo = sp.diff(ceq,_k)
        fk = sp.diff(ceq,x)
        
        bessel = {'besselj':jn,'besselk':kn,'sqrt':np.sqrt}
        libraries = [bessel, "numpy"] 
        
        V = self._k*a*np.sqrt(self._n**2-1)
        if V > 2.405:
            print("WARNING:Multimode regime. V = ",V)
            raise ValueError 
        
        foo_eq = sp.lambdify([x,_k],sp.re(ceq),modules=libraries)
        eig_eq = lambda x: foo_eq(x,self._k)
        
        """ First boundary condition (characteristic equation) """
        self._beta = krlv(eig_eq, self._k*(self._n+1)/2)
        #self._beta = brd(eig_eq, self._k*(self._n+1)/2, f_tol = 0.0001)

        del ha,qa
        """
        _______________________________________
        Group velocity, Evaluate h and q not as functions of beta
        _______________________________________
        """
  
        
        self.vg = sp.lambdify([x,_k],-fk/fo, modules=libraries)(self._beta,self._k)
        ha = a*np.sqrt(self._n*self._n*self._k*self._k - self._beta*self._beta)
        qa = a*np.sqrt(self._beta*self._beta - self._k*self._k)
        
        
        """
        
        Mode components as it presented in Balykin's paper. 
        Cylindrical components are exact as in fundamental_mode.py
        Decomposition components are multiplied by sqrt(2) (discuss!! It was my fault)
        The polariztion components of the mode are:
        E- =  E e- + dE exp(-2i phi) e+ + Ez exp(-i phi) ez

        To get the opponent polarisation:
        'r$ E_{-\sigma}^{\mu} = (E_{\sigma}^{-\mu})^{\asr} $'
        
        It is normalized to be $ \int |E(r)| ^ 2 dx dy = \lambdabar ^ 2 (x,y in cm), 
                            or  \int |E(r)| ^ 2 dx dy = 1 (x,y in \lambdabar) $ 
        """

        s = (1/qa/qa+1/ha/ha)/((-kn(0,qa)-kn(2,qa))/2/qa/kn(1,qa)+(jn(0,ha)-jn(2,ha))/2/ha/jn(1,ha))
        
        efin =  lambda r :-1 * self._beta / 2 / ha * a * (1-s) * jn(0,ha * r /a) * np.sqrt(2) / jn (1,ha)
        efout = lambda r :-1 * self._beta / 2 / qa * a * (1-s) * kn(0,qa * r /a) * np.sqrt(2) / kn (1,qa)
        
        E = lambda r: np.piecewise(r, [r>=a,r<a], [efout,efin])
        
        dfin =  lambda r :-1* self._beta / 2 / ha * a * (1+s) * jn(2,ha * r /a) * np.sqrt(2) / jn (1,ha)
        dfout = lambda r : 1 * self._beta / 2 / qa * a * (1+s) * kn(2,qa * r /a) * np.sqrt(2) / kn (1,qa)
        
        dE = lambda r: np.piecewise(r, [r>=a,r<a], [dfout,dfin])
        
        zfin =  lambda r : jn(1,ha * r /a)  / jn (1,ha)
        zfout = lambda r :kn(1,qa * r /a) / kn (1,qa)
        
        Ez = lambda r: np.piecewise(r, [r>=a,r<a], [zfout,zfin])


        
        
        _fooint = lambda r:  r*(abs(E(r))**2 + abs(dE(r))**2 + abs(Ez(r))**2)
        _norm = np.sqrt((quad(_fooint,a,np.inf)[0]+n*n*quad(_fooint,0,a)[0])*2*np.pi)

        print('A=', 1/_norm)
        self.E = lambda r:  1j*E(r) / _norm
        self.dE = lambda r: 1j*dE(r) /_norm
        self.Ez = lambda r: Ez(r)/_norm
        
        self.Ephi = lambda r: 1j / np.sqrt(2) * (self.E(r) + self.dE(r))
        self.Er = lambda r: 1 / np.sqrt(2) * (self.E(r) - self.dE(r))

        #H-field

        eps = n*n

        hzi = lambda r: self._beta*s*jn(1,ha * r/a) / jn(1,ha) / _norm
        hzo = lambda r: self._beta*s*kn(1,ha * r/a) / kn(1,ha) / _norm

        self.Hz = lambda r: 1j * np.piecewise(r, [r>=a,r<a], [hzo,hzi])

        hfi = lambda r: ((eps - s*(self._beta)**2) * jn(0,ha * r/a) - (eps + s*(self._beta)**2) * jn(2,ha *r/a) ) * a / 2 / ha / jn(1,ha) / _norm
        hfo = lambda r: ((1 - s*(self._beta)**2) * kn(0,qa * r/a) + (1 + s*(self._beta)**2) * kn(2,qa *r/a) ) * a / 2 / qa / kn(1,qa) / _norm

        self.Hphi = lambda r: 1j * np.piecewise(r, [r>=a,r<a], [hfo,hfi])

        hri = lambda r: ((eps - s*(self._beta)**2) * jn(0,ha * r/a) + (eps + s*(self._beta)**2) * jn(2,ha *r/a) ) * a / 2 / ha / jn(1,ha) / _norm
        hro = lambda r: ((1 - s*(self._beta)**2) * kn(0,qa * r/a) - (1 + s*(self._beta)**2) * kn(2,qa *r/a) ) * a / 2 / qa / kn(1,qa) / _norm

        self.Hr = lambda r: 1 * np.piecewise(r, [r>=a,r<a], [hro,hri])

        """
        xr = np.linspace(0.01, 5, 100)
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        plt.plot(xr, abs(self.Hr(xr)), 'b', label = r'$|H_{\rho}|$')
        plt.plot(xr, abs(self.Hphi(xr)), 'r', label =  r'$|H_{\phi}|$')
        plt.plot(xr, abs(self.Hz(xr)), 'g', label = r'$|H_{z}|$')
        plt.legend()
        plt.show()

        #full mode: time and z dependance

        def vecE(rho,phi,t):
            return np.asarray([self.Er(rho)*np.exp(1j*phi)*np.exp(-1j*t), -self.Ephi(rho)*np.exp(1j*phi)*np.exp(-1j*t), self.Ez(rho)*np.exp(1j*phi)*np.exp(-1j*t)], dtype=complex)

        def vecH(rho,phi,t):
            return np.asarray([-self.Hr(rho)*np.exp(1j*phi)*np.exp(-1j*t), self.Hphi(rho)*np.exp(1j*phi)*np.exp(-1j*t), -self.Hz(rho)*np.exp(1j*phi)*np.exp(-1j*t)], dtype=complex)

        #full poynting vector (exact and time-averaged)
        def vecS(rho,phi,t):
            A = ([[np.cos(phi), -np.sin(phi), 0.],
                  [np.sin(phi), np.cos(phi), 0.],
                  [0.,0.,1.]])

            return np.dot(A,np.cross(np.real(vecE(rho,phi,t)), np.real(vecH(rho,phi,t)))) / (4*np.pi)

        def vecSta(rho,phi):
            t = 0
            A = ([[np.cos(phi), -np.sin(phi), 0.],
                  [np.sin(phi), np.cos(phi), 0.],
                  [0.,0.,1.]])

            return np.real(np.dot(A,np.cross((vecE(rho,phi,t)), np.conjugate(vecH(rho,phi,t)))) / (8*np.pi))

        denEni = lambda rho: eps* np.linalg.norm(vecE(rho,0,0),axis=0) + np.linalg.norm(vecH(rho,0,0),axis=0)
        denEno = lambda rho: np.linalg.norm(vecE(rho,0,0),axis=0) + np.linalg.norm(vecH(rho,0,0),axis=0)

        denE = lambda r: np.piecewise(r, [r >= a, r < a], [denEno, denEni])

        import matplotlib.cm as cm
        x,y = np.meshgrid(np.linspace(-2*a, 2*a), np.linspace(-2*a, 2*a))

        fig = plt.figure()
        ax = fig.gca()
        circle2 = plt.Circle((0, 0), a, color='k', fill=False)
        ax.add_artist(circle2)
        z = denE(np.sqrt(x**2+y**2))
        ax.imshow(z, interpolation='gaussian', cmap=cm.get_cmap('jet'),
                origin='lower', extent=[-2*a, 2*a, -2*a, 2*a],
                vmax=abs(z).max()*1.1, vmin=-abs(z).min())


        plt.show()


        output_dir = "Poynting_vector"

        """

        """
        #uncomment to precession
        time = np.linspace(0,2*np.pi,50)
        for t in time:

            soa = np.array([[ 0,  0, 0.,     *vecS(0,0,t)],
                            [ 0,  0.5*a, 0., *vecS(0.5*a ,np.pi/2 ,t)],
                            [ 0.5*a,  0, 0., *vecS(0.5*a ,0       ,t)],
                            [ 0, -0.5*a, 0., *vecS(0.5*a ,-np.pi/2,t)],
                            [-0.5*a,  0, 0., *vecS(0.5*a ,np.pi   ,t)],

                            [ 0.5 * a/np.sqrt(2),  0.5 * a/np.sqrt(2), 0., *vecS(0.5 * a, np.pi / 4, t)],
                            [ 0.5 * a/np.sqrt(2), -0.5 * a/np.sqrt(2), 0., *vecS(0.5 * a, -np.pi / 4, t)],
                            [-0.5 * a/np.sqrt(2), 0.5 * a/np.sqrt(2), 0., *vecS(0.5 * a, 3*np.pi / 4, t)],
                            [-0.5 * a/np.sqrt(2), -0.5 * a/np.sqrt(2), 0., *vecS(0.5 * a, -3*np.pi/4, t)],

                            [1.5 * a / np.sqrt(2), 1.5 * a / np.sqrt(2), 0., *vecS(1.5 * a, np.pi / 4, t)],
                            [1.5 * a / np.sqrt(2), -1.5 * a / np.sqrt(2), 0., *vecS(1.5 * a, -np.pi / 4, t)],
                            [-1.5 * a / np.sqrt(2), 1.5 * a / np.sqrt(2), 0., *vecS(1.5 * a, 3 * np.pi / 4, t)],
                            [-1.5 * a / np.sqrt(2), -1.5 * a / np.sqrt(2), 0., *vecS(1.5 * a, -3 * np.pi / 4, t)],

                            [ 0,  1.5*a, 0., *vecS(1.5*a, np.pi / 2, t)],
                            [ 1.5*a,  0, 0., *vecS(1.5*a, 0, t)],
                            [ 0, -1.5*a, 0., *vecS(1.5*a, -np.pi / 2, t)],
                            [-1.5*a,  0, 0., *vecS(1.5*a, np.pi, t)]])

            X, Y, Z, U, V, W = zip(*soa)
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.set_xlim([1.498*a/np.sqrt(2), 1.502*a/np.sqrt(2)])
            ax.set_ylim([1.498*a/np.sqrt(2), 1.502*a/np.sqrt(2)])
            ax.set_zlim([0, 0.005])
            ax.quiver(X, Y, Z, U, V, W)
            ax.view_init(90,0)
            plt.savefig(output_dir + '/snapshot_%1.2f.png' % t, dpi=100)


        os.system("convert -delay 0.5 -dispose Background +page " + str(output_dir) + \
          "/*.png -loop 0 " + str(output_dir) + "/animation.gif")

        """




if __name__ == '__main__':
    args = {'k':1, 
            'n':1.45,
            'a': 2*np.pi*200/780
            }
    m = exact_mode(**args)
    c = 2*np.pi*200/780 * 1.5

