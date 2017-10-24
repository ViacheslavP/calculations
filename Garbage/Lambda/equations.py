import numpy as np
from scipy.special import jn, kn


n = 1.45
eps = n * n
a = 2*np.pi*200/780

def chareq(k,lam, omega=1.):

    eps_l = 1 + lam*(eps - 1)
    #probing limits
    
    
    ha = np.sqrt(eps_l * omega * omega - k * k) * a
    qa = np.sqrt(k * k - omega * omega) * a
    kd = (-kn(0, qa) / 2 - kn(2, qa) / 2) * qa / kn(1, qa)
    jd = (jn(0, ha) / 2 - jn(2, ha) / 2) * ha / jn(1, ha)

    ceq = ((jd * eps_l * ( qa / ha) ** 2 + kd) * (jd * (qa / ha) ** 2 + kd)) - (
        ((eps_l - 1) * k * omega * a * a / ha / ha) ** 2)

    return ceq