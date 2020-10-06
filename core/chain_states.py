# methods and procedures for creating various chains
class atomic_state(object):
    def __init__(self, noa: int, zpos, campl):
        if (len(zpos) != noa) or (len(campl) != noa):
            raise TypeError("Wrong Atomic State")
        self.noa = noa
        self.zpos = np.asarray(zpos, dtype=np.float64)
        if not np.vdot(campl, campl) == 0:
            self.campl = campl / np.vdot(campl, campl)
        else:
            self.campl = campl


# Creating a simple chain of noa atoms with period d
def create_chain(noa: int, d, random=False) -> object:
    zpos = d * np.arange(noa)
    if random:
        zpos = noa * d * np.sort(np.random.rand(noa))
    campl = np.ones_like(zpos, dtype=np.complex)
    return atomic_state(noa, zpos, campl)


# Creating an excited chain that decays mostly into waveguide
def create_excited_chain(noa: int, d) -> object:
    chain = create_chain(noa, d)
    for i in range(noa):
        chain.campl[i] *= np.exp(-2j * np.pi * chain.zpos[i])
    return chain


# Creating an chain that has no excitation
def create_mirror_like_chain(noa: int, d, random=False) -> object:
    chain = create_chain(noa, d, random)
    chain.campl = np.zeros_like(chain.campl, dtype=np.complex)
    return chain


# Merging different chains
def merge_atomic_states(astate: object, bstate: object, distance, to_end=True) -> object:
    n1, n2 = astate.noa, bstate.noa
    add_dist = 0
    if to_end:
        add_dist = astate.zpos[-1]
    zpos = np.concatenate((astate.zpos, bstate.zpos + distance + add_dist), axis=None)
    campl = np.concatenate((astate.campl, np.exp(2j * (distance + add_dist) * np.pi) * bstate.campl), axis=None)
    return atomic_state(n1 + n2, zpos, campl)
