
import numpy as np
import scipy as sp
import os
from collections import Iterable

class Hamiltonian:
    def __init__(self, logp, grad, start, step_size=1, n_steps=5):
        self.start = start
        self.step_size = step_size/(len(self.start))**(1/4)
        self.n_steps = n_steps
        self.logp = logp
        self.grad=grad
        self.state = start


    def step(self):
        q = self.state
        p = self.draw_momentum()
        y, r = q, p
        for i in range(self.n_steps):
            y, r = leapfrog(y, r, self.step_size, self.grad)
        if accept(q, y, p, r, self.logp):
            q = y
            self._accepted += 1
        self.state = q
        self._sampled += 1
        return x

    @property
    def acceptance_rate(self):
        return self._accepted/self._sampled

    def flatten(items):
        for x in items:
            if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
                for sub_x in flatten(x):
                    yield sub_x
            else:
                yield x

    def leapfrog(self,q, p, step_size, grad):
        p = p + step_size/2*grad(x)
        q = q + step_size*p
        p = p + step_size/2*grad(x1)
        return q, p


    def accept(self,q, y, p, r):
        E_new = energy(y, r)
        E = energy(q, p)
        A = np.min(np.array([0, E_new - E]))
        return (np.log(np.random.rand()) < A)


    def energy(self,logp, q, p):
        pf=list(flatten(p.values()))
        return self.logp(q) - 0.5*np.dot(pf, pf)


    def draw_momentum(self):
        momentum = {}
        for var in self.start:
            dim=(np.array(self.start[var])).size
            if dim==1:
                momentum[var]=np.random.normal(0,1)
            else:
                mass_matrix=np.identity(dim)
                momentum[var]=np.random.multivariate_normal(np.zeros(dim), mass_matrix)
        return momentum


    def sample(niter=1e3):
        for i in range(niter):
            self.step()
