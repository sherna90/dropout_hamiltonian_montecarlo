
import numpy as np
import scipy as sp
import os
from utils import *
from numpy.linalg import inv
from copy import deepcopy
from multiprocessing import Pool
import os 

def unwrap_self_mcmc(arg, **kwarg):
    return HMC.sample(*arg, **kwarg)

class HMC:
    def __init__(self, X,y,logp, grad, start,hyper, n_steps=5,scale=True,transform=True,verbose=True):
        self.start = start
        self.hyper = hyper
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
        #epsilon=direction*0.2
        q = deepcopy(self.state)
        p = self.draw_momentum()
        print('process : %d, q : %s , p : %s '%(os.getpid(),q,p) )
        q_new=deepcopy(q)
        p_new=deepcopy(p)
        for i in range(self.n_steps):
            q_new, p_new = self.leapfrog(q_new, p_new, epsilon)
        if self.accept(q, q_new, p, p_new):
            q = deepcopy(q_new)
            p = p_new
            self._accepted += 1
        self.state = deepcopy(q)
        self.momentum=p.copy()
        self._sampled += 1
        return self.state

    def acceptance_rate(self):
        return float(self._accepted)/self._sampled

    def leapfrog(self,q, p,epsilon):
        grad_q=self.grad(self.X,self.y,q,self.hyper)
        for var in self.start.keys():
            p[var] = p[var] + (epsilon/2.)*grad_q[var]
            q[var] = q[var] + epsilon*self._inv_mass_matrix[var].dot(p[var].reshape(-1)).reshape(self.start[var].shape)
        grad_q_new=self.grad(self.X,self.y,q,self.hyper)
        for var in self.start.keys():
            p[var] = p[var] + (epsilon/2.)*grad_q_new[var]
        return q, p

    def accept(self,current_q, proposal_q, current_p, proposal_p):
        E_new = self.energy(proposal_q, proposal_p)
        E = self.energy(current_q, current_p)
        A = np.exp(E - E_new)
        g = np.random.rand()
        return (g < A)


    def energy(self, q, p):
        U=0
        for var in self.start.keys():
            U+=0.5*np.dot(p[var].reshape(-1).T,self._inv_mass_matrix[var]).dot(p[var].reshape(-1))
        return -self.logp(self.X,self.y,q,self.hyper) + U


    def draw_momentum(self):
        momentum={}
        for var in self.start.keys():
            dim=(np.array(self.start[var])).size
            if dim==1:
                momentum[var]=np.random.normal(0,self._mass_matrix[var])
            else:
                mass_matrix=self._mass_matrix[var]
                momentum[var]=np.random.multivariate_normal(np.zeros(dim), mass_matrix).reshape(self.start[var].shape)
        return momentum


    def sample(self,niter=1e4,burnin=1e3):
        for i in range(int(niter)):
            if i>burnin and (i%(niter/10)==0):
                self.compute_mass_matrix(int(burnin),False)
            self._samples.append(self.step())
            if self._verbose and (i%(niter/10)==0):
                print('process : %d, acceptance rate : %s '%(os.getpid(),self.acceptance_rate()) )
        posterior={}
        for var in self.start.keys():
            posterior[var]=[]
        for s in self._samples[int(burnin):]:
            for var in self.start.keys():
                posterior[var].append(s[var].reshape(-1))
        for var in self.start.keys():
            posterior[var]=np.array(posterior[var])
        return posterior

    def multicore_sample(self,niter=1e4,burnin=1e3,ncores=8):
        pool = Pool(processes=ncores)
        results=pool.map(unwrap_self_mcmc, zip([self]*4, [int(niter/ncores)]*ncores,[burnin]*ncores))
        posterior={}
        for var in self.start.keys():
            posterior[var]=np.concatenate([r[var] for r in results],axis=0)
        return posterior

    def compute_mass_matrix(self,burnin,cov=True):
        alpha=0.9
        n=len(self._samples)
        posterior={}
        for var in self.start.keys():
            posterior[var]=[]
        for s in self._samples[int(burnin):]:
            for var in self.start.keys():
                posterior[var].append(s[var].reshape(-1))
        for var in self.start.keys():
            posterior[var]=np.array(posterior[var])
            if cov:
                self._mass_matrix[var]=alpha*np.cov(posterior[var].T)+(1.0-alpha)*np.identity((np.array(self.start[var])).size)
                self._inv_mass_matrix[var]=inv(self._mass_matrix[var])
            else:
                self._mass_matrix[var]=np.var(posterior[var],axis=0)*np.identity((np.array(self.start[var])).size)
                self._inv_mass_matrix[var]=inv(self._mass_matrix[var])
            
        
            
