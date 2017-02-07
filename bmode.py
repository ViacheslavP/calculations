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
        self._qa = 0.
        self._ha = 0.
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
        self._ha = a*np.sqrt(self._n*self._n*self._k*self._k - self._beta*self._beta)
        self._qa = a*np.sqrt(self._beta*self._beta - self._k*self._k) 
        
        
        """
        
        Mode components as it presented in Balykin's paper. 
        Cylindrical components are exact as in fundamental_mode.py
        Decomposition components are multiplied by sqrt(2) (discuss!! It was my fault)
        The polariztion components of the mode are: 
        E+  =  E e+ + dE exp(2 i phi) e- + Ez exp(i phi) ez
        E- =  - E e- - dE exp(-2i phi) e+ + Ez exp(-i phi) ez
        
        It is normalized to be $ \int |E(r)| ^ 2 dx dy = \lambdabar ^ 2 (x,y in cm), 
                            or  \int |E(r)| ^ 2 dx dy = 1 (x,y in \lambdabar) $ 
        """
        
        ha = self._ha
        qa = self._qa
        s = (1/qa/qa+1/ha/ha)/((-kn(0,qa)-kn(2,qa))/2/qa/kn(1,qa)+(jn(0,ha)-jn(2,ha))/2/ha/jn(1,ha))
        
        efin =  lambda r :-1 * self._beta / 2 / ha * a * (1-s) * jn(0,ha * r /a) * np.sqrt(2) / jn (1,ha)
        efout = lambda r :-1 * self._beta / 2 / qa * a * (1-s) * kn(0,qa * r /a) * np.sqrt(2) / kn (1,qa)
        
        E = lambda r: np.piecewise(r, [r>=a,r<a], [efout,efin])
        
        dfin =  lambda r :1* self._beta / 2 / ha * a * (1+s) * jn(2,ha * r /a) * np.sqrt(2) / jn (1,ha)
        dfout = lambda r :-1 * self._beta / 2 / qa * a * (1+s) * kn(2,qa * r /a) * np.sqrt(2) / kn (1,qa)
        
        dE = lambda r: np.piecewise(r, [r>=a,r<a], [dfout,dfin])
        
        zfin =  lambda r : jn(1,ha * r /a)  / jn (1,ha)
        zfout = lambda r :kn(1,qa * r /a) / kn (1,qa)
        
        Ez = lambda r: np.piecewise(r, [r>=a,r<a], [zfout,zfin])
        
        
        
        _fooint = lambda r: r* (abs(E(r))**2 + abs(dE(r))**2 + abs(Ez(r))**2)
        _norm = np.sqrt(quad(_fooint,0,np.inf)[0]*2*np.pi)
        
        self.E = lambda r: -1j*E(r) / _norm
        self.dE = lambda r: 1j*dE(r) /_norm
        self.Ez = lambda r: Ez(r)/_norm
        
        self.Ephi = lambda r: -1j / np.sqrt(2) * (self.E(r) + self.dE(r))
        self.Er = lambda r: -1 / np.sqrt(2) * (self.E(r) - self.dE(r))        

if __name__ == '__main__':
    args = {'k':1, 
            'n':1.4469,
            'a': 2*np.pi*200/850
            }
    m = exact_mode(**args)
    print(m._beta)
