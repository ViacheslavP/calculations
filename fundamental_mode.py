# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 13:40:45 2016

@author: viacheslav
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
        self.generate_mode()
        
        
    def generate_mode(self):
        """
        1.Due to the fact that characteristic equation contains various bessel
        functions, I am using symbolical calculus. The alter way is to use 
        high order approximation of derivatives.
        ceq(x,_k) = 0 - the spectral problem equation
        x = beta
        _k = omega/c
        2.In multimode regime some sort of bias could happen. 
        3.Since I got several problems with straight solving boundary conditions,
        I prefer to use symbolical solver
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
        

        
        r,hc = sp.symbols('r,hc')
        ha = self._ha
        qa = self._qa
        
        
        """
        _______________________________________
        Not yet normalized mode amplitudes
        _______________________________________
        """
        
        """The In-field"""
        _ein = sp.besselj(1,ha*r/a)/sp.besselj(1,ha)
        _hin = hc*sp.besselj(1,ha*r/a)/sp.besselj(1,ha)
        _rin = -1j*a*a/ha/ha * (self._beta*sp.diff(_ein,r) + self._k*1j*_hin/r)
        _ain = -1j*a*a/ha/ha * (self._beta*1j*_ein/r - self._k*sp.diff(_hin,r))
        
        """The Out-field"""
        _eout = sp.besselk(1,qa*r/a)/sp.besselk(1,qa)
        _hout = hc*sp.besselk(1,qa*r/a)/sp.besselk(1,qa)
        _rout = 1j*a*a/qa/qa * (self._beta*sp.diff(_eout,r)+self._k*1j*_hout/r)
        _aout = 1j*a*a/qa/qa * (self._beta*1j*_eout/r - self._k*sp.diff(_hout,r))
        

        """ Second boundary condition """        
        for t in sp.solveset(sp.Eq(_rout,n*n*_rin),hc).subs({r:a}):
            hc_e = t.evalf()
            
        _hin = _hin.subs({hc:hc_e})
        _hout = _hout.subs({hc:hc_e})
        _rin = _rin.subs({hc:hc_e})
        _ain = _ain.subs({hc:hc_e})
        _rout = _rout.subs({hc:hc_e})
        _aout = _aout.subs({hc:hc_e})
        
        
      
        """ Noramlization """
        _in_dense = (r*(_rin*sp.conjugate(_rin)+_ain*sp.conjugate(_ain)+_ein*sp.conjugate(_ein)))
        _out_dense = (r*(_rout*sp.conjugate(_rout)+_aout*sp.conjugate(_aout)+_eout*sp.conjugate(_eout)))
        _in = sp.lambdify(r,_in_dense,modules=libraries)
        _out = sp.lambdify(r,_out_dense,modules=libraries)
        
        _norm = np.sqrt((quad(_in,0,a)[0]+quad(_out,a,np.inf)[0])*2*np.pi)
        
        """
        _______________________________________
        Normalized mode amplitudes
        _______________________________________
        """
        _ein = 1j*_ein/_norm
        _eout = 1j*_eout/_norm
        _rin = 1j*_rin/_norm
        _ain = 1j*_ain/_norm
        _rout = 1j*_rout/_norm
        _aout = 1j*_aout/_norm
        
        
        """ 
        ________________________________________
        Approximation and plotting 
        ________________________________________
        
        """
        
        """
        \vec E = (E_r, E_phi, E_z) - components of fundamental mode in 
        cylindrical coordinate system. 
        Analytical form is represented in Klein et al. 2004.
        
        E and dE - amplitudes of field in polarization plane (decomposition components)
        E stands for amplitude of m=0 irreduceble component of SO(2) (E ~ Bessel0)
        dE stands for amplitude of m=+-2 ... (dE ~ Bessel2)
        See modes.pdf for decomposition
        """
            
        rin = sp.lambdify(r,sp.re(_rin),modules=libraries)
        rout = sp.lambdify(r,sp.re(_rout),modules=libraries)
        ain = sp.lambdify(r,_ain,modules=libraries)
        aout = sp.lambdify(r,_aout,modules=libraries)
        ein = sp.lambdify(r,_ein,modules=libraries)
        eout = sp.lambdify(r,_eout,modules=libraries)    
        
        E_phi = lambda r: np.piecewise(r, [r>=a,r<a], [aout,ain])
            
        E_r = lambda r: np.piecewise(r, [r>=a,r<a], [rout,rin])
            
        E_z = lambda r: np.piecewise(r, [r>=a,r<a], [eout,ein])
            
        Ein = sp.lambdify(r,sp.re(_rin-1j*_ain)*0.5,modules=libraries)
        Eout = sp.lambdify(r,sp.re(_rout-1j*_aout)*0.5,modules=libraries)
        dEin = sp.lambdify(r,sp.re(_rin+1j*_ain)*0.5,modules=libraries)
        dEout = sp.lambdify(r,sp.re(_rout+1j*_aout)*0.5,modules=libraries)
        
        E = lambda r: np.piecewise(r, [r>=a,r<a], [Eout,Ein])

        dE = lambda r: np.piecewise(r, [r>=a,r<a], [dEout,dEin])        
        
        """
        dE -> 0, E_z -> 0 - paraxial approx.
        Changing of normalization: int |\vec E | ^2 = int |E|^2 
        Gauss Approximations of E outside of waveguide  
        """
        ga = lambda x,w: 1/np.sqrt(np.pi*w**2)*np.exp(-x**2/w**2/2)            
        _relp = lambda x: x*E(x)*E(x)        
        _norm2 = np.sqrt(quad(_relp,0,np.inf)[0]*2*np.pi)
        relp = lambda x: E(x)/_norm2 
        xb = np.linspace(0.1*a,1*a,100)
        t,v = cf(ga,xb,relp(xb),bounds = (0,100))
        self.wb = t[0]
        
        from matplotlib import rc, rcParams
        
        font = {'family' : 'serif',
                'weight' : 'book',
                'size'   : 12}
        rc('font', **font)
        rcParams.update({'figure.autolayout': True})
        if True:
            """Plotting exact components"""
            rin = sp.lambdify(r,sp.Abs(_rin),modules=libraries)
            rout = sp.lambdify(r,sp.Abs(_rout),modules=libraries)
            ain = sp.lambdify(r,sp.Abs(_ain),modules=libraries)
            aout = sp.lambdify(r,sp.Abs(_aout),modules=libraries)
            ein = sp.lambdify(r,sp.Abs(_ein),modules=libraries)
            eout = sp.lambdify(r,sp.Abs(_eout),modules=libraries) 
            
            E_phi = lambda r: np.piecewise(r, [r>=a,r<a], [aout,ain])
                
            E_r = lambda r: np.piecewise(r, [r>=a,r<a], [rout,rin])
                
            E_z = lambda r: np.piecewise(r, [r>=a,r<a], [eout,ein])
            
            from matplotlib import pyplot as plt
            xr = np.linspace(0.1*a, 5*a, 150)
            yrin = np.linspace(0.01*a, 0.97*a, 50)
            yout = np.linspace(1.03*a, 5*a, 100)
            
            plt.plot(yrin/a, abs(E_r(yrin)), 'r', label = '$E_r (r)$',lw=1.5)
            plt.plot(yout/a, abs(E_r(yout)),'r',lw=1.5)
            plt.plot(xr/a,abs(E_phi(xr)), label = '$-i E_{\phi} (r)$', lw=1.5)
            plt.plot(xr/a, abs(E_z(xr)),label = '$-i E_z(r)$',lw = 1.5)
            plt.axvline(x=1.0, ymin=-0.05, ymax=1.,color='k',ls='dashed',label='Waveguide \n boundary')
            #plt.legend(frameon=False,bbox_to_anchor=(0.55, 1), loc=2, borderaxespad=0.)
            plt.xlabel('$r, a$',fontsize=16)
            plt.ylabel('$HE_{11}$'+' mode \n [arb. units]',fontsize=16)
            plt.savefig('Exact components 1.45.svg',dpi=700)
            plt.show()
            
        if True:
            """Plotting decomposition components"""

            ein = sp.lambdify(r,sp.im(_ein),modules=libraries)
            eout = sp.lambdify(r,sp.im(_eout),modules=libraries) 
            
            E =  lambda r: np.piecewise(r, [r>=a,r<a], [Eout,Ein])
            dE =  lambda r: np.piecewise(r, [r>=a,r<a], [dEout,dEin])
            E_z = lambda r: np.piecewise(r, [r>=a,r<a], [eout,ein])
            from matplotlib import pyplot as plt

            xr = np.linspace(0.05*a, 5*a, 100)
            yin = np.linspace(0.05*a,0.97*a,50)
            yout = np.linspace(1.03*a,5*a)
            
            plt.plot(yin/a,Ein(yin), 'r-',lw=1.5,label='$E(r)$')
            plt.plot(yout/a, Eout(yout),'r-',lw=1.5)
            
            plt.plot(yin/a,dEin(yin),'b-',lw=1.5,label='$\delta E(r)$')
            plt.plot(yout/a, dEout(yout),'b-',lw=1.5)
            
            plt.plot(xr/a,E_z(xr),'g-',lw=1.5, label = '$-iE_z(r)$')
            plt.plot(xr/a,_norm2*ga(xr,self.wb),'k',lw=2, label = 'Gaussian')
            plt.axvline(x=1.0, ymin=-0.05, ymax=1.,color='k',ls='dashed',label='Waveguide \n boundary')
            #plt.legend(frameon=False,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            #plt.legend(frameon=False)           
            plt.xlabel('$r, a$',fontsize=16)
            plt.ylabel('$HE_{11}$'+' decomposition amplitudes \n [arb. units]',fontsize=16)
            plt.savefig('Decomposition 1.45.svg',dpi=700)
            plt.show()
            
        if False:
            """Plotting approximations"""
            from matplotlib import pyplot as plt
            xr = np.linspace(0.1*a,5*a,100)
            yin = np.linspace(0.1*a,0.97*a,50)
            yout = np.linspace(1.03*a, 5*a,100)
            plt.plot(yin/a,relp(yin), 'g', label = '$E_{pa}(r)$',lw=2)
            plt.plot(yout/a,relp(yout), 'g',lw=2)
            plt.plot(xr/a,ga(xr,self.wb),'m',lw=2, label = 'Gaussian')
            plt.axvline(x=1.0, ymin=-0.05, ymax=1.,color='k',ls='dashed',label='Waveguide \n boundary')
            #plt.title('Fundamental mode gaussian fit',fontsize=14)
            plt.xlabel('$r,a$',fontsize=16)
            plt.ylabel('$E_{pa}$'+', Gaussian fit',fontsize=16)
            plt.legend(frameon=False, bbox_to_anchor=(1.05, 1))
            #plt.text(2.5,0.13,r'$\frac{1}{ \sqrt{\pi w^2}} \times  exp(-r^2/{2w^2})$',fontsize=16)
            plt.text(3.0,0.13, '$w=%02.2f a$'% (self.wb/a),fontsize=16)
            plt.savefig('Gaussian fit n=1.45.svg',dpi=400)
            plt.show()
            
        if not os.path.isfile("mode_data_%g_%g_%g" % (self._k,self._n,self._a)):
            with open("mode_data_%g_%g_%g" % (self._k,self._n,self._a),'w') as f:
                ar = np.asarray([self._beta,self.wb,self.vg])
                ar.tofile(f)
            
if __name__ == '__main__':
    a= 2*np.pi*200/780
    k = 1.
    n = 1.45
    m = exact_mode(k,n,a)