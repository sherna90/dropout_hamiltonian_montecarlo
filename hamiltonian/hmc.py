
import numpy as np
import scipy as sp
import os
from utils import *
from numpy.linalg import inv
from copy import deepcopy
from multiprocessing import Pool
from tqdm import tqdm, trange
import os 

def unwrap_self_mcmc(arg, **kwarg):
    return HMC.sample(*arg, **kwarg)

class HMC:
    def __init__(self, X,y,logp, grad, start,hyper, path_length=4,scale=True,transform=True,verbose=True):
        self.start = start
        self.hyper = hyper
        self.step_size = {}
        self.path_length = path_length
        self.logp = logp
        self.grad=grad
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
        self._direction=1
        self._mass_matrix={}
        self._inv_mass_matrix={}
        for var in self.start.keys():
            dim=(np.array(self.start[var])).size
            self.step_size[var]=(1./dim)**(0.25)
            if dim==1:
                self._mass_matrix[var]=1
                self._inv_mass_matrix[var]=1
            else:
                self._mass_matrix[var]=np.identity(dim)
                self._inv_mass_matrix[var]=np.identity(dim)
        self._verbose=verbose


    def step(self,state,momentum):
        path_length = np.random.rand() * self.path_length
        n_steps = max(1, int(path_length / max(self.step_size.values())))
        #direction = 1.0 if np.random.random() > 0.5 else -1.0
        q = deepcopy(state)
        p = self.draw_momentum()
        #p=deepcopy(momentum)
        #print('process : %d, q : %s , p : %s '%(os.getpid(),q,p) )
        q_new=deepcopy(q)
        p_new=deepcopy(p)
        for i in range(n_steps):
            q_new, p_new = self.leapfrog(q_new, p_new, self._direction)
        if self.accept(q, q_new, p, p_new):
            q = deepcopy(q_new)
            p = p_new
            self._accepted += 1
        else:
            self._direction=-1.0
        return q,p

    def acceptance_rate(self):
        return float(self._accepted)/len(self._samples)

    def leapfrog(self,q, p,direction):
        grad_q=self.grad(self.X,self.y,q,self.hyper)
        for var in self.start.keys():
            epsilon=direction*self.step_size[var]
            p[var] = p[var] - (0.5*epsilon)*grad_q[var]
            q[var] = q[var] + epsilon*self._inv_mass_matrix[var].dot(p[var].reshape(-1)).reshape(self.start[var].shape)
            grad_q=self.grad(self.X,self.y,q,self.hyper)
            p[var] = p[var] - (0.5*epsilon)*grad_q[var]
        return q, p

    def accept(self,current_q, proposal_q, current_p, proposal_p):
        E_new = self.energy(proposal_q, proposal_p)
        E = self.energy(current_q, current_p)
        accept_stat = min(1, np.exp(E - E_new))
        g = np.random.rand()
        return (g <= accept_stat)


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


    def sample(self,niter=1e4,warmup=1e2,burnin=1e3):
        total_iter=int(warmup+burnin)
        q,p=self.start,self.draw_momentum()
        for i in tqdm(range(total_iter),total=total_iter):
            q,p=self.step(q,p)
            if i>int(warmup):
                self._samples.append(q)
        self.compute_mass_matrix(True)
        q,p=self.start,self.draw_momentum()
        while self._samples:
            self._samples.pop()
        self._accepted = 0
        print('process : %d, warmup done! '%os.getpid() )
        for i in tqdm(range(int(niter)),total=int(niter)):
            q,p=self.step(q,p)
            self._samples.append(q)
            if self._verbose and (i%(niter/10)==0):
                print('process : %d, acceptance rate : %s '%(os.getpid(),self.acceptance_rate()) )
        posterior={}
        for var in self.start.keys():
            posterior[var]=[]
        for s in self._samples:
            for var in self.start.keys():
                posterior[var].append(s[var].reshape(-1))
        for var in self.start.keys():
            posterior[var]=np.array(posterior[var])
        return posterior

    def multicore_sample(self,niter=1e4,warmup=1e2,burnin=1e3,ncores=4):
        pool = Pool(processes=ncores)
        results=pool.map(unwrap_self_mcmc, zip([self]*4, [int(niter/ncores)]*ncores,[burnin]*ncores))
        posterior={}
        for var in self.start.keys():
            posterior[var]=np.concatenate([r[var] for r in results],axis=0)
        return posterior

    def compute_mass_matrix(self,cov=True):
        alpha=0.9
        n=len(self._samples)
        posterior={}
        for var in self.start.keys():
            posterior[var]=[]
        for s in self._samples:
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
                print(var)
                print(self._mass_matrix[var]) 
        self._momentum=self.draw_momentum()
            
        
            
