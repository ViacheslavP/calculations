import numpy as np
from ens_class import ensemble, gd, nb, displacement, step      
from matplotlib import rc, rcParams
font = {'family' : 'serif',
        'weight' : 'book',
        'size'   : 12}
rc('font', **font)
rcParams.update({'figure.autolayout': True})

def plot_NRT(alpha = 0., num = 20, env = 'fiber', addit = ''):

    deltaP = np.arange(-1, 1, 0.01)*gd
    nsp = len(deltaP);
    r = []
    t = []
    x= range(5,num,1)
    for i in x:
        chi = ensemble(env,'chain',dist = alpha,d=5.5)
        chi.generate_ensemble()
        t.append(chi.Transmittance[0])
        r.append(chi.Reflection[0])
    from matplotlib import pyplot as plt
    
    if chi._env == 'fiber' :
        s = '$Beam\, waist = %g \lambda,\,  \lambda ,\, Fiber\, ON$' % (alpha)
    elif chi._env == 'vacuum':
        s = '$Beam\, waist = %g \lambda,\, \lambda, \, Fiber\, OFF$' % (alpha)
    else:
        raise NameError('Got problem with fiber')
    plt.subplots(figsize=(10,7))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle(s)
    
    axr = plt.subplot(211)
    axr.plot(x,[(abs(i)**2) for i in r])
    axr.set_ylabel('$(|r|^2)$')
    axr.ticklabel_format(style='plain')
    
    axt = plt.subplot(212)
    axt.plot(x,[abs(i)**2 for i in t])
    axt.set_ylabel('$(|t|^2)$')
    axt.ticklabel_format(style='plain')
    axt.set_xlabel('Number of atoms')
    plt.savefig(("RTN_%g_%s" % (alpha,chi._env))+addit+'.png')
    plt.show()

def plot_Bragg(st,to,N=5,neighbours=4,condition='chain',disp=0.):
    from matplotlib import pyplot as plt
    dK = np.linspace(st,to,120)*2*np.pi
    t = [];r = []
    for dk in dK:
        chi = ensemble(N,
                       neighbours,
                       'fiber',
                       condition,
                       dist = disp,
                       d=1.5,
                       l0=dk,
                       deltaP=np.linspace(-1,1,100)
                       )
               
        chi.generate_ensemble()
        t.append(np.max(abs(chi.Transmittance)))
        r.append(np.max(abs(chi.Reflection)))
        
    m =ensemble(N,
                       neighbours,
                       'fiber',
                       condition,
                       dist = disp,
                       d=1.5,
                       l0=2*np.pi/1.09518,
                       deltaP=np.linspace(-10,10,100))
    m.generate_ensemble()
                   
    f, (ax, ax2) = plt.subplots(2, 1, sharex=True)
    ax.plot(dK / 2 / np.pi, [(i**2) for i in t], color = 'g', lw = 1.5, label = 'Transmittance')
    ax.plot(dK / 2 / np.pi, [(i**2) for i in r], color = 'm', lw = 1.5,label = 'Reflectance')
    ax.legend(frameon=False)
    #ax2.plot(dK / 2 / np.pi, [(i**2) for i in t], color = 'g', lw = 1.5, label = 'Transmittance')
    ax2.plot(dK / 2 / np.pi, [(i**2) for i in r], color = 'm', lw = 1.5,label = 'Reflection')
    ax.set_ylim(.83, 0.9)  # outliers only
    ax2.set_ylim(0, .29)  # most of the data

    
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop='off')  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()


    d = .015  # how big to make the diagonal lines in axes coordinates

    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    ax.set_ylabel('$T$')
    ax2.set_ylabel('$R$')
    plt.xlabel('$d/ \lambda$',fontsize=16)
    plt.savefig('cascade.svg')
    plt.show()


   
def plot_Bragg_family(st,to,N=5,neighbours=4,condition='chain',disp=0.,ty='L'):
        
    from matplotlib import pyplot as plt
    P = np.linspace(st,to,120)
    beta = 1.0951842440279331
    e = ensemble(N,
                   neighbours,
                   'fiber',
                   condition,
                   dist = disp,
                   d=1.5,
                   l0=(1)*np.pi / beta,
                   deltaP=P,
                   typ=ty)
    m1 = ensemble(N,
                   neighbours,
                   'fiber',
                   condition,
                   dist = disp,
                   d=1.5,
                   l0=(1-0.05)*np.pi / beta,
                   deltaP=P,
                   typ=ty)
    m2 = ensemble(N,
                   neighbours,
                   'fiber',
                   condition,
                   dist = disp,
                   d=1.5,
                   l0=(1-0.075)*np.pi / beta,
                   deltaP=P,
                   typ=ty)
    p1 = ensemble(N,
                   neighbours,
                   'fiber',
                   condition,
                   dist = disp,
                   d=1.5,
                   l0=(1+0.05)*np.pi / beta,
                   deltaP=P,
                   typ=ty)
    p2 = ensemble(N,
                   neighbours,
                   'fiber',
                   condition,
                   dist = disp,
                   d=1.5,
                   l0=(1+0.075)*np.pi / beta,
                   deltaP=P,
                   typ=ty)
                   
    e.generate_ensemble()
    m1.generate_ensemble()
    m2.generate_ensemble()
    p1.generate_ensemble()
    p2.generate_ensemble()
    
    
    plt.title('Reflection spectra', fontsize = 16)
    plt.xlabel('Detuning, '+'$\gamma$', fontsize=16)
    plt.ylabel('Reflectance', fontsize=16)
    plt.plot(P,(abs(e.Reflection)**2), color = 'k', lw=2., label = "$\Delta \lambda_m = 0$")
    
    plt.plot(P,(abs(m1.Reflection)**2),color = 'b', lw = 1.2, label = '$\Delta \lambda_m = -0.05 \lambda_m$')
    plt.plot(P,(abs(p1.Reflection)**2),color = 'r', lw = 1.2,  label = '$\Delta \lambda_m = 0.05 \lambda_m$')
    
    plt.plot(P,(abs(m2.Reflection)**2),color = 'g',lw = 1.2, label = '$\Delta \lambda_m = -0.075 \lambda_m$')    
    plt.plot(P,(abs(p2.Reflection)**2),color = 'm',lw = 1.2, label = '$\Delta \lambda_m = 0.075 \lambda_m$')
    
    plt.axvline(x=0, ymin=0.06,color='k',ls='dashed',label='Atomic \n resonance \n frequency')   
    plt.legend(frameon=False,bbox_to_anchor=(0.50, 0.95), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.subplots_adjust(right= 0.7)
    plt.savefig('RefBragg.svg',dpi=300)
    plt.show()
    
    
    plt.title('Transmission spectra', fontsize = 16)
    plt.xlabel('Detuning, '+'$\gamma$', fontsize=16)
    plt.ylabel('Transmittance', fontsize=16)
    plt.plot(P,(abs(e.Transmittance)**2), color = 'k',  lw=2.5, label = "$\Delta \lambda_m = 0$")
    
    #plt.plot(P,(abs(m1.Transmittance)**2),color = 'b', lw = 1.2, label = '$\Delta \lambda = -0.05 \lambda$')
    #plt.plot(P,(abs(p1.Transmittance)**2),color = 'r', lw = 1.2,  label = '$\Delta \lambda = 0.05 \lambda$')
    
    plt.plot(P,(abs(m2.Transmittance)**2),color = 'g',lw = 1.5, label = '$\Delta \lambda_m = -0.075 \lambda_m$')    
    plt.plot(P,(abs(p2.Transmittance)**2),color = 'm',lw = 1.5, label = '$\Delta \lambda_m = 0.075 \lambda_m$')
    
    plt.axvline(x=0, ymin=0.06, ymax=1.,color='k',ls='dashed',label='Atomic \n resonance \n frequency')   
    plt.legend(frameon=False,bbox_to_anchor=(0.55, 0.7), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.subplots_adjust(right= 0.7)
    plt.savefig('TranBragg.svg',dpi=300)    
    plt.show()
    

def plot_O_vs_DO(st,to, N=5,neighbours=4,condition='chain'):
    from matplotlib import pyplot as plt
    freq = np.linspace(st,to,100)
    la = ensemble(N,
                   neighbours,
                   'fiber', #Stands for cross-section calculation(vacuum) or 
                             #transition calculations (fiber) 
                   'chain',  #Stands for atom positioning
                   dist = 0.,
                   d=1.5,
                   l0=step,
                   deltaP=freq,
                   typ = 'L',
                   dens=1.)
                   
    lb = ensemble(N,
                   neighbours,
                   'fiber', #Stands for cross-section calculation(vacuum) or 
                             #transition calculations (fiber) 
                   'chain',  #Stands for atom positioning
                   dist = 0.1*2*np.pi,
                   d=1.5,
                   l0=step,
                   deltaP=freq,
                   typ = 'L',
                   dens=1.)
                   
                
                   
    la.generate_ensemble()
    lb.generate_ensemble()
    
    plt.plot(freq,(np.log(abs(la.Reflection)**2)), color = 'k', lw=2., label = "$\Delta \lambda = 0$")
    plt.plot(freq,(np.log(abs(lb.Reflection)**2)), color = 'b', lw=2., label = "$\Delta \lambda = 0$")
     
    plt.show()
    
    plt.title('Reflection spectra', fontsize = 16)
    plt.xlabel('Detuning, '+'$\gamma$', fontsize=16)
    plt.ylabel('ln Reflectance', fontsize=16)
    plt.plot(freq,np.log(abs(la.Reflection)**2), color = 'g', lw=1.2, label = "$\sigma = 0 \lambda$")
    
    plt.plot(freq,np.log(abs(lb.Reflection)**2),color = 'm', lw = 1.2, label = '$\sigma = 0.1 \lambda$')
    
    plt.axvline(x=0, ymin=0.06,color='k',ls='dashed',label='Atomic \n resonance \n frequency')   
    plt.legend(frameon=False,bbox_to_anchor=(0.55, 1.00), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.subplots_adjust(right= 0.7)
    plt.savefig('RefOvs.svg',dpi=300)
    plt.show()
    
    
    plt.title('Transmission spectra', fontsize = 16)
    plt.xlabel('Detuning, '+'$\gamma$', fontsize=16)
    plt.ylabel('ln Transmittance', fontsize=16)
   
    
    plt.plot(freq,np.log(abs(la.Transmittance)**2), color = 'g', lw=1.5, label = "$\sigma = 0 \lambda$")
    
    plt.plot(freq,np.log(abs(lb.Transmittance)**2),color = 'm', lw = 1.5, label = '$\sigma = 0.1 \lambda$')
    
    plt.axvline(x=0, ymin=0.06, ymax=1.,color='k',ls='dashed',label='Atomic \n resonance \n frequency')   
    plt.legend(frameon=False,bbox_to_anchor=(0.55, 0.7), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.subplots_adjust(right= 0.7)
    plt.savefig('TranOvs.svg',dpi=300)    
    plt.show()
                   
                   
def plot_V_vs_L(st,to,N=4,neighbours=3,condition='chain',disp=0.):
    from matplotlib import pyplot as plt
    P = np.linspace(st,to,120)
    eL = ensemble(N,
                neighbours,
                'fiber',
                condition,
                dist = disp,
                d=1.5,
                l0=2*np.pi/1.0,
                deltaP=P,
                typ='L')
    eV = ensemble(N,
                neighbours,
                'fiber',
                condition,
                dist = disp,
                d=1.5,
                l0=2*np.pi/1.0,
                deltaP=P,
                typ='V')
                
    eL.generate_ensemble()
    eV.generate_ensemble()
    
    plt.title('Reflection spectra', fontsize = 16)
    plt.xlabel('Detuning, '+'$\gamma$', fontsize=16)
    plt.ylabel('Reflectance', fontsize=16)
    
    
    plt.plot(P,(abs(eL.Reflection)**2),color = 'b', lw = 1.2, label = '$\Lambda$')
    plt.plot(P,(abs(eV.Reflection)**2),color = 'r', lw = 1.2,  label = '$V$')
    
    
    plt.axvline(x=0, ymin=-16,color='k',ls='dashed',label='Atomic \n resonance \n frequency')   
    plt.legend(frameon=False,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.subplots_adjust(right= 0.7)
    plt.savefig('RefBragg.svg',dpi=300)
    plt.show()
    
def plot_Lamb_Shift(start,stop,N=5,neighbours=4,condition='chain',disp=0.):
    
    from matplotlib import pyplot as plt
    dK = np.linspace(start,stop,240)
    dP = np.linspace(-4,4,70)
    
    rm = []
    for k in dK:
        
        e = ensemble(N,
                     neighbours,
                     'fiber',
                     condition,
                     dist = 0.,
                     d=1.5,
                     l0=k,
                     deltaP=dP)
        e.generate_ensemble()        
        rm.append(dP[np.argmax(abs(e.Reflection)**2)])
        
        
    plt.plot(dK,rm,'r',lw=2)
    plt.xlabel('Step, '+'$ \lambda$',fontsize=16)
    plt.ylabel('Collective Lamb Shift, '+ '$\Delta_{CL}, \gamma$',fontsize=16)
    plt.savefig('Lamb.svg')
    plt.show()
        
        
                     
def AverageCrsc(eps=1e-2, na=5, dens=0.5, N=20,freq = np.linspace(-15,15,200)):
    

    ACrSec = np.zeros(len(freq))
    W=0
    for i in range(N):
        
        chi = ensemble(na,
               na-1,
               'vacuum',
               'chain',
               dist = displacement,
               d=1.5,
               l0=step,
               deltaP=freq,
               typ = 'L',
               dens = dens)
               
        chi.generate_ensemble()
        wi = 1 
        ACrSec += wi*np.real(chi.CrSec) 
        W += wi
        chi.vis_crsec()        
        
    import matplotlib.pyplot as plt
    plt.plot(freq, ACrSec/W, 'b', lw =2.)
    plt.xlabel('Detuning, '+'$\gamma$', fontsize=16)
    plt.ylabel('$ \sigma  $', fontsize=16)
    plt.savefig('Average %g.png' % dens)
    plt.show()
    freq[np.argmax(ACrSec)]
    return ACrSec/W

    
"""
_____________________________________________________________________________
Declaring global variables
_____________________________________________________________________________

System of units:

Distance is measured in \lambdabar (lambd)
Time is measured in 1/ {\gamma} (g)
Speed of light is the only scale constant: c = lambd * g  = g / omega = T/ {\tau}
IDK what \hbar is for.

"""


#chi.visualize()


#plot_Bragg_family(-2,2)

"""
m = exact_mode(1,1.45,1)
m.generate_mode()
ha = m._ha
qa = m._qa
s = m._s

k = qa/ha * (((1-s)*j0(ha)-(1+s)*jn(2,ha))/((1-s)*k0(qa)+(1+s)*kn(2,qa)))*(k1(qa)/j1(ha)) - 1/n0/n0
k12 = qa/ha * (((1-s)*j0(ha)+(1+s)*jn(2,ha))/((1-s)*k0(qa)-(1+s)*kn(2,qa)))*(k1(qa)/j1(ha))

print(k,k12)

"""

"""
d = []
nf = [1.30,1.35,1.40,1.45,1.50,1.55,1.60,1.65,1.70]
for i in nf:
    m = exact_mode(2*np.pi,i)
    d.append(m.wb)

n = 1.45

import matplotlib.pyplot as plt
m = exact_mode(c/lambd*2*np.pi,1.45, 1.)
m.generate_mode()

cx = np.vectorize(m.cx)
cy = np.vectorize(m.cy)

y = np.arange(-3*a,3*a,0.4*a)
x = np.arange(-3*a,3*a,0.4*a)

X,Y = np.meshgrid(x,y)
U = cx(X,Y).real
W = cy(X,Y).real
plt.quiver(X,Y,U,W)
plt.show()

phi = np.linspace(0,2*np.pi,100)
plt.polar(phi, ex(1.5*a,phi,0))
plt.polar(phi, ey(1.5*a,phi,0)) 
plt.show()
si = t[0]
x = np.arange(0, a*5, 0.01)
f = lambda y: (1/np.sqrt(np.pi*si**2)*np.exp(-y**2/si**2/2))
g = [f(i) for i in x]
plt.plot(x,ey(x),'m')
plt.plot(x,g)
plt.plot(x,ez(x), 'r')
plt.show()

wb = t[0]
chi = ensemble('fiber','chain',dist=displacement)
chi.generate_ensemble()
chi.visualize()

"""
"""
if __name__ == "__main__":

    t1 = time()

    deltaP = np.arange(-15, 20.1, 0.1)*gd
    nsp = len(deltaP);    
    
    wb = 4.
    chi = ensemble('vacuum','chain',dist = displacement)
    chi.generate_ensemble()
    chi.visualize()
    
    wb = 25.
    chi = ensemble('vacuum','chain',dist = displacement)
    chi.generate_ensemble()
    chi.visualize()
    
    wb = 50.
    chi = ensemble('vacuum','chain',dist = displacement)
    chi.generate_ensemble()
    chi.visualize()
    
    del chi
    
    wb = 4.
    vac = ensemble('vacuum','chain',0.); vac.generate_ensemble(); vac.visualize('unitarity')
    fib = ensemble('fiber','chain',0.); fib.generate_ensemble(); fib.visualize('unitarity')
     
    del vac,fib
    wb = 50.
    plot_NRT(5., 200, env = 'vacuum')
    plot_NRT(0., 200, env = 'vacuum')
    
    print('Executed for ', (time()-t1)/60, 'm')
    
"""

