
import numpy as np
import scipy as sp
import os
from utils import *

class HMC:
    def __init__(self, X,y,logp, grad, start, n_steps=5,scale=True,transform=True,verbose=True):
        self.start = start
        self.step_size = 1./n_steps
        self.n_steps = n_steps
        self.logp = logp
        self.grad=grad
        self.state = start
        self.momentum=self.draw_momentum()
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
        self.verbose=verbose


    def step(self):
        direction = np.random.choice([-1, 1], p=[0.5, 0.5])
        epsilon=direction*self.step_size*(1.0+np.random.normal(1))
        q = self.state.copy()
        p = self.draw_momentum()
        grad_q=self.grad(self.X,self.y,q)
        q_new=q.copy()
        p_new=p.copy()
        p_new['bias'] = p_new['bias'] + (epsilon/2)*grad_q['bias']
        p_new['weights'] = p_new['weights'] + (epsilon/2)*grad_q['weights']
        q_new['bias'] = q['bias'] + epsilon*p_new['bias']
        q_new['weights'] = q_new['weights'] + epsilon*p_new['weights']
        for i in range(self.n_steps-1):
            q_new, p_new = self.leapfrog(q_new, p_new, epsilon)
        p_new['bias'] = p_new['bias'] + (epsilon/2)*grad_q['bias']
        p_new['weights'] = p_new['weights'] + (epsilon/2)*grad_q['weights']
        q_new['bias'] = q['bias'] + epsilon*p_new['bias']
        q_new['weights'] = q_new['weights'] + epsilon*p_new['weights']
        grad_q=self.grad(self.X,self.y,q_new)
        if self.accept(q, q_new, p, p_new):
            q = q_new
            p = p_new
            self._accepted += 1
        self.state = q.copy()
        self.momentum=p.copy()
        self._sampled += 1
        return self.state

    def acceptance_rate(self):
        return float(self._accepted)/self._sampled

    def leapfrog(self,q, p,epsilon):
        grad_q=self.grad(self.X,self.y,q)
        p['bias'] = p['bias'] + epsilon*grad_q['bias']
        p['weights'] = p['weights'] + epsilon*grad_q['weights']
        q['bias'] = q['bias'] + epsilon*p['bias']
        q['weights'] = q['weights'] + epsilon*p['weights']
        return q, p


    def accept(self,q, y, p, r):
        E_new = self.energy(y, r)
        E = self.energy(q, p)
        A = np.min(np.array([0, E_new - E]))
        return (np.log(np.random.rand()) < A)


    def energy(self, q, p):
        U=0.5*np.dot(p['weights'].reshape(-1).T,p['weights'].reshape(-1))
        U+=0.5*np.dot(p['bias'].T,p['bias'])
        return -self.logp(self.X,self.y,q) - U


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
            if self.verbose and (i%(niter/10)==0):
                print('acceptance rate : {0:.4f}'.format(self.acceptance_rate()) )
            
