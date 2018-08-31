
import numpy as np
import scipy as sp
import os
from utils import *
from numpy.linalg import inv
from copy import deepcopy

class MH:
    def __init__(self, X,y,logp,start,hyper,scale=True,transform=True,verbose=True):
        self.start = start
        self.hyper = hyper
        self.logp = logp
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
        self._verbose=verbose
        

    def step(self):
        q = self.state.copy()
        q_new=deepcopy(q)
        self.random_walk(q_new)
        if self.accept(q, q_new):
            q = q_new
            self._accepted += 1
        self.state = q.copy()
        self._sampled += 1
        return self.state

    def acceptance_rate(self):
        return float(self._accepted)/self._sampled


    def accept(self,current_q, proposal_q):
        E_new = self.energy(proposal_q)
        E = self.energy(current_q)
        A = np.exp(E - E_new)
        g = np.random.rand()
        return (g < A)


    def energy(self, q):
        return -self.logp(self.X,self.y,q,self.hyper)


    def random_walk(self,q):
        for var in self.start.keys():
            dim=(np.array(self.start[var])).size
            if dim==1:
                q[var]+=np.random.normal(0,1)
            else:
                mass_matrix=np.identity(dim)
                q[var]+=np.random.multivariate_normal(np.zeros(dim), mass_matrix).reshape(self.start[var].shape)
        return q


    def sample(self,niter=1e3,burnin=100):
        for i in range(int(niter)):
            self._samples.append(self.step())
            if self._verbose and (i%(niter/10)==0):
                print('acceptance rate : {0:.4f}'.format(self.acceptance_rate()) )
        posterior={}
        for var in self.start.keys():
            posterior[var]=[]
        for s in self._samples[int(burnin):]:
            for var in self.start.keys():
                posterior[var].append(s[var].reshape(-1))
        for var in self.start.keys():
            posterior[var]=np.array(posterior[var])
        return posterior

            
        
            
