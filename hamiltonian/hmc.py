import numpy as np
import scipy as sp
import os
from utils import *
from numpy.linalg import inv
from copy import deepcopy
from multiprocessing import Pool
from tqdm import tqdm, trange
import h5py 
import os 
from scipy.optimize import check_grad
import math 

def unwrap_self_mcmc(arg, **kwarg):
    return HMC.sample(*arg, **kwarg)

class HMC:
    def __init__(self, X,y,logp, grad, start,hyper, path_length=None,step_size=None,verbose=True):
        self.X=X
        self.y=y
        self.start = start
        self.hyper = hyper
        self.step_size = step_size if step_size is not None else 1
        self.path_length = path_length if path_length is not None else 2 * math.pi
        self.logp = logp
        self.grad=grad
        self._accepted=0
        self._direction=1.0
        self._mass_matrix={}
        self._inv_mass_matrix={}
        for var in self.start.keys():
            dim=(np.array(self.start[var])).size
            if dim==1:
                self._mass_matrix[var]=np.array(1.0)
                self._inv_mass_matrix[var]=np.array(1.0)
            else:
                self._mass_matrix[var]=np.ones(dim)
                self._inv_mass_matrix[var]=np.ones(dim)
        self._verbose=verbose


    def step(self,state,momentum,rng):
        n_steps =max(1, int(self.path_length / self.step_size))
        direction = 1.0 if rng.rand() > 0.5 else -1.0
        epsilon={var:direction*self.step_size for var in self.start.keys()}
        q = deepcopy(state)
        p = self.draw_momentum(rng)
        q_new, p_new = self.leapfrog(q, p, epsilon,n_steps)
        accepted=self.accept(q, q_new, p, p_new,rng) 
        if accepted:
            q = q_new
            p = p_new
            self._accepted += 1 
        return q,p,accepted

    def acceptance_rate(self,accepted,samples):
        return np.sum(1.0*np.array(accepted))/len(samples)

    def leapfrog(self,q, p,epsilon,n_steps):
        q_new=deepcopy(q)
        p_new=deepcopy(p)
        grad_q=self.grad(self.X,self.y,q_new,self.hyper)
        for i in range(n_steps):   
            for var in self.start.keys():
                p_new[var]-= (0.5*epsilon[var])*grad_q[var]
                q_new[var]+=epsilon[var]*self._inv_mass_matrix[var].reshape(self.start[var].shape)*p_new[var]
            grad_q=self.grad(self.X,self.y,q_new,self.hyper)
            for var in self.start.keys():
                p_new[var]-= (0.5*epsilon[var])*grad_q[var]
        return q_new, p_new

    def accept(self,current_q, proposal_q, current_p, proposal_p,rng):
        accept=False
        E_new = self.energy(proposal_q,proposal_p)
        E_current = self.energy(current_q,current_p)
        #print(E_new,E_current)
        A = np.exp(E_current - E_new)
        #print('p:',proposal_p,current_p)
        #print('q:',proposal_q,current_q)
        g = rng.rand()
        #print(g,A)
        if np.isfinite(A) and (g < A):
            accept=True
        return accept


    def potential_energy(self,p):
        K=0
        for var in self.start.keys():
            K+=0.5*np.sum(self._inv_mass_matrix[var].reshape(self.start[var].shape)*np.square(p[var]))
        return K

    def energy(self, q, p):
        K=self.potential_energy(p)
        U=self.logp(self.X,self.y,q,self.hyper)
        return K+U 


    def draw_momentum(self,rng):
        momentum={}
        for var in self.start.keys():
            dim=(np.array(self.start[var])).size
            momentum[var]=rng.normal(0,self._inv_mass_matrix[var],dim)
            momentum[var]=momentum[var].reshape(self.start[var].shape)
        return momentum


    def sample(self,niter=1e4,burnin=1e3,backend=None,rng=None):
        samples=[]
        burnin_samples=[]
        accepted=[]
        if rng==None:
            rng = np.random.RandomState()
        q,p=self.start,self.draw_momentum(rng)
        for i in tqdm(range(int(burnin))):
            q,p,a=self.step(q,p,rng)
            burnin_samples.append(q)
            accepted.append(a)
            if i==burnin-1: 
                self.compute_mass_matrix(burnin_samples,alpha=0.9)
                acc_rate=self.acceptance_rate(accepted,burnin_samples)
                #self.step_size=self.tune(self.step_size,acc_rate)
                print('burnin acceptance rate : {0:.4f}'.format(acc_rate))
                del accepted[:]
                del burnin_samples[:]
        for i in tqdm(range(int(niter))):
            q,p,a=self.step(q,p,rng)
            accepted.append(a)
            samples.append(q)
            if self._verbose and (i%(niter/10)==0):
                print('acceptance rate : {0:.4f}'.format(self.acceptance_rate(accepted,samples)) )
        posterior={var:[] for var in self.start.keys()}
        for s in samples:
            for var in self.start.keys():
                posterior[var].append(s[var].reshape(-1))
        for var in self.start.keys():
            posterior[var]=np.array(posterior[var])
        #return posterior
        #if backend is None:
        #    for i in tqdm(range(int(niter)),total=int(niter)):
        #        q,p=self.step(q,p)
        #        self._samples.append(q)
        #        if self._verbose and (i%(niter/10)==0):
        #            print('acceptance rate : {0:.4f}'.format(self.acceptance_rate()) )
        #    posterior={var:[] for var in self.start.keys()}
        #    for s in self._samples:
        #        for var in self.start.keys():
        #            posterior[var].append(s[var].reshape(-1))
        #    for var in self.start.keys():
        #        posterior[var]=np.array(posterior[var])
        #else:
        #    posterior=h5py.File(backend,'w')
        #    num_samples=int(niter)
        #    dset = {var:posterior.create_dataset(var, (num_samples,self.start[var].reshape(-1).shape[0]), maxshape=(None,self.start[var].reshape(-1).shape[0]) ) for var in self.start.keys()}
        #    for i in tqdm(range(int(niter)),total=int(niter)):
        #        q,p=self.step(q,p)
        #        for var in self.start.keys():
        #            dset[var][-1,:]=q[var].reshape(-1)
        #    posterior.flush()
        return posterior

    def multicore_sample(self,niter=1e4,burnin=1e3,backend=None,ncores=2):
        pool = Pool(processes=ncores)
        rng = [np.random.RandomState(i) for i in range(ncores)]
        results=pool.map(unwrap_self_mcmc, zip([self]*ncores, [int(niter/ncores)]*ncores,[burnin]*ncores,[backend]*ncores,rng))
        posterior={var:np.concatenate([r[var] for r in results],axis=0) for var in self.start.keys()}
        return posterior

    def compute_mass_matrix(self,samples,alpha=0.9):
        posterior={var:[] for var in self.start.keys()}
        for s in samples:
            for var in self.start.keys():
                posterior[var].append(s[var].reshape(-1))
        for var in self.start.keys():
            posterior[var]=np.array(posterior[var])
            self._mass_matrix[var]=alpha*self._mass_matrix[var]+(1-alpha)*np.var(posterior[var],axis=0)
            self._inv_mass_matrix[var]=1./self._mass_matrix[var]
            
    def tune(self,scale,acc_rate):
            new_scale=scale
            if acc_rate < 0.001:
                print('reduce by 90 percent')
                new_scale *= 0.1
            elif acc_rate < 0.05:
                print('reduce by 50 percent')
                new_scale *= 0.5
            elif acc_rate < 0.2:
                print('reduce by ten percent')
                # reduce by ten percent
                new_scale *= 0.9
            elif acc_rate > 0.95:
                print('increase by factor of ten')
                # increase by factor of ten
                new_scale *= 10.0
            elif acc_rate > 0.75:
                print('increase by double')
                new_scale *= 2.0
            elif acc_rate > 0.5:
                print('increase by ten percent')
                # increase by ten percent
                new_scale *= 1.1
            return new_scale      