
import numpy as np
import scipy as sp
import os
from utils import *
from numpy.linalg import inv
from copy import deepcopy
from multiprocessing import Pool
import os 

def unwrap_self_mcmc(arg, **kwarg):
    return SGHMC.sample(*arg, **kwarg)

class SGHMC:
    def __init__(self, X,y,logp, grad, start,hyper,alpha, n_steps=5,scale=True,transform=True,verbose=True):
        self.start = start
        self.hyper = hyper
        self.alpha = alpha
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


    def step(self,X_batch,y_batch,state,momemtum):
        direction = np.random.choice([-1, 1], p=[0.5, 0.5])
        epsilon=direction*self.step_size*(1.0+np.random.normal(1))
        #epsilon=direction*0.2
        q = deepcopy(state)
        p = deepcopy(momemtum)
        print('process : %d, q : %s , p : %s '%(os.getpid(),q,p) )
        q_new=deepcopy(q)
        p_new=deepcopy(p)
        for i in range(self.n_steps):
            q_new, p_new = self.leapfrog(q_new, p_new, epsilon,X_batch,y_batch)
        self._sampled += 1
        return (q_new, p_new)

    def acceptance_rate(self):
        return float(self._accepted)/self._sampled

    def leapfrog(self,q, p,epsilon,X_batch,y_batch):
        for var in self.start.keys():
            q[var]+=p[var]
        grad_q=self.grad(X_batch,y_batch,q,self.hyper)
        for var in self.start.keys():    
            p[var] =(1-self.alpha)*p[var] + (epsilon)*grad_q[var]
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


    def sample(self,niter=1e4,burnin=1e3,batch_size=20):
        (state,momemtum)=(self.state,self._momentum)
        for i in range(int(niter)):
            for batch in self.iterate_minibatches(self.X, self.y, batch_size):
                X_batch, y_batch = batch
                if i>burnin and (i%(niter/10)==0):
                    self.compute_mass_matrix(int(burnin),False)
                (state,momemtum)=self.step(X_batch,y_batch,state,momemtum)
                self._samples.append(state)
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
            
    def iterate_minibatches(self,X, y, batchsize):
        assert X.shape[0] == y.shape[0]
        for start_idx in range(0, X.shape[0] - batchsize + 1, batchsize):
            excerpt = slice(start_idx, start_idx + batchsize)
            yield X[excerpt], y[excerpt]
