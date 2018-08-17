
import numpy as np
import scipy as sp
import os
from utils import *
from numpy.linalg import inv

class HMC:
    def __init__(self, X,y,logp, grad, start, n_steps=5,scale=True,transform=True,verbose=True):
        self.start = start
        self.step_size = 1./n_steps
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
        self._samples=[]
        self._accepted=0
        self._sampled=0
        self._mass_matrix={}
        self._inv_mass_matrix={}
        for var in self.start.keys():
            dim=(np.array(self.start[var])).size
            if dim==1:
                self._mass_matrix[var]=1
                self._inv_mass_matrix[var]=1
            else:
                self._mass_matrix[var]=np.identity(dim)
                self._inv_mass_matrix[var]=np.identity(dim)
        self._verbose=verbose
        self._momentum=self.draw_momentum()


    def step(self):
        direction = np.random.choice([-1, 1], p=[0.5, 0.5])
        epsilon=direction*self.step_size*(1.0+np.random.normal(1))
        q = self.state.copy()
        p = self.draw_momentum()
        q_new=q.copy()
        p_new=p.copy()
        for i in range(self.n_steps):
            q_new, p_new = self.leapfrog(q_new, p_new, epsilon)
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
        p['bias'] = p['bias'] + (epsilon/2.)*grad_q['bias']
        p['weights'] = p['weights'] + (epsilon/2.)*grad_q['weights']
        q['bias'] = q['bias'] + epsilon*self._inv_mass_matrix['bias'].dot(p['bias'].reshape(-1)).reshape(self.start['bias'].shape)
        q['weights'] = q['weights'] + epsilon*self._inv_mass_matrix['weights'].dot(p['weights'].reshape(-1)).reshape(self.start['weights'].shape)
        grad_q_new=self.grad(self.X,self.y,q)
        p['bias'] = p['bias'] + (epsilon/2.)*grad_q_new['bias']
        p['weights'] = p['weights'] + (epsilon/2.)*grad_q_new['weights']
        return q, p

    def accept(self,q, y, p, r):
        E_new = self.energy(y, r)
        E = self.energy(q, p)
        A = np.min(np.array([0, E_new - E]))
        return (np.log(np.random.rand()) < A)


    def energy(self, q, p):
        U=0.5*np.dot(p['weights'].reshape(-1).T,self._inv_mass_matrix['weights']).dot(p['weights'].reshape(-1))
        U+=0.5*np.dot(p['bias'].T,self._inv_mass_matrix['bias']).dot(p['bias'])
        return -self.logp(self.X,self.y,q) - U


    def draw_momentum(self):
        momentum={'weights':np.zeros((self.start['weights'].shape)),'bias':np.zeros((self.start['bias'].shape))}
        for var in momentum.keys():
            dim=(np.array(self.start[var])).size
            if dim==1:
                momentum[var]=np.random.normal(0,self._mass_matrix[var])
            else:
                mass_matrix=self._mass_matrix[var]
                momentum[var]=np.random.multivariate_normal(np.zeros(dim), mass_matrix).reshape(self.start[var].shape)
        return momentum


    def sample(self,niter=1e3,burnin=100):
        for i in range(int(niter)):
            if i>burnin and (i%(niter/10)==0):
                self.compute_mass_matrix(int(burnin))
            self._samples.append(self.step())
            if self._verbose and (i%(niter/10)==0):
                print('acceptance rate : {0:.4f}'.format(self.acceptance_rate()) )
        bias_data=[]
        weights_data=[]
        for s in self._samples[int(burnin):]:
            bias_data.append(s['bias'].reshape(-1))
            weights_data.append(s['weights'].reshape(-1))
        bias_data=np.array(bias_data)
        weights_data=np.array(weights_data)
        return bias_data,weights_data

    def compute_mass_matrix(self,burnin):
        alpha=0.5
        n=len(self._samples)
        bias_data=[]
        weights_data=[]
        for s in self._samples[burnin:]:
            bias_data.append(s['bias'].reshape(-1))
            weights_data.append(s['weights'].reshape(-1))
        bias_data=np.array(bias_data)
        weights_data=np.array(weights_data)
        self._mass_matrix['bias']=alpha*np.cov(bias_data.T)+(1.0-alpha)*np.identity((np.array(self.start['bias'])).size)
        self._inv_mass_matrix['bias']=inv(self._mass_matrix['bias'])
        self._mass_matrix['weights']=alpha*np.cov(weights_data.T)+(1.0-alpha)*np.identity((np.array(self.start['weights'])).size)
        self._inv_mass_matrix['weights']=inv(self._mass_matrix['weights'])

        
            
