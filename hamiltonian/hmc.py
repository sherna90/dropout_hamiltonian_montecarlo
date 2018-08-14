
import numpy as np
import scipy as sp
import os
from utils import *

class HMC:
    def __init__(self, X,y,logp, grad, start, step_size=1, n_steps=5,scale=True,transform=True,verbose=True):
        self.start = start
        self.step_size = step_size/(len(self.start))**(1/4)
        self.n_steps = n_steps
        self.logp = logp
        self.grad=grad
        self.state = start
        if scale:
            self.X,x_min,x_max=scaler_fit(X[:])
        else:
            self.X=X
        if transform:
            classes=np.unique(y)
            self.y=one_hot(y,len(classes))
        else: 
            self.y=y
        self.samples=[]
        self._accepted=0
        self._sampled=0


    def step(self):
        q = self.state
        p = self.draw_momentum()
        y, r = q.copy(), p.copy()
        for i in range(self.n_steps):
            y, r = self.leapfrog(y, r)
        if self.accept(q, y, p, r):
            q = y
            self._accepted += 1
        self.state = q
        self._sampled += 1
        return self.state

    def acceptance_rate(self):
        return float(self._accepted)/self._sampled

    def leapfrog(self,q, p):
        grad_q=self.grad(self.X,self.y,q)
        p['bias'] = p['bias'] + self.step_size/2*grad_q['bias']
        p['weights'] = p['weights'] + self.step_size/2*grad_q['weights']
        q['bias'] = q['bias'] + self.step_size*p['bias']
        q['weights'] = q['weights'] + self.step_size*p['weights']
        grad_q=self.grad(self.X,self.y,q)
        p['bias'] = p['bias'] + self.step_size/2*grad_q['bias']
        p['weights'] = p['weights'] + self.step_size/2*grad_q['weights']
        return q, p


    def accept(self,q, y, p, r):
        E_new = self.energy(y, r)
        E = self.energy(q, p)
        A = np.min(np.array([0, E_new - E]))
        return (np.log(np.random.rand()) < A)


    def energy(self, q, p):
        p_w=[p[k] for k in p.keys() if k!='alpha']
        p_f=list(flatten(p_w))
        return self.logp(self.X,self.y,q) - 0.5*np.dot(p_f, p_f)


    def draw_momentum(self):
        momentum={'weights':np.zeros((self.start['weights'].shape)),'bias':np.zeros((self.start['bias'].shape))}
        for var in momentum.keys():
            dim=(np.array(self.start[var])).size
            if dim==1:
                momentum[var]=np.random.normal(0,1)
            else:
                mass_matrix=np.identity(dim)
                momentum[var]=np.random.multivariate_normal(np.zeros(dim), mass_matrix).reshape(self.start[var].shape)
        return momentum


    def sample(self,niter=1e3):
        for i in range(int(niter)):
            self.samples.append(self.step())
            if i%(niter/10)==0:
                #print self.acceptance_rate()
                print('acceptance rate : {0:.4f}'.format(self.acceptance_rate()) )
            
