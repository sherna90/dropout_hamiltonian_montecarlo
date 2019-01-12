import numpy as np
import scipy as sp
import os
from utils import *
from numpy.linalg import inv
from copy import deepcopy
from multiprocessing import Pool,cpu_count
from tqdm import tqdm, trange
import h5py 
import os 
from scipy.optimize import check_grad
import math 

def unwrap_self_mcmc(arg, **kwarg):
    return HMC.sample(*arg, **kwarg)


class HMC:
    def __init__(self, X,y,logp, grad, start,hyper, path_length=None,verbose=True):
        self.X=X
        self.y=y
        self.start = start
        self.hyper = hyper
        self.step_size = 1.0
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
        #n_steps =max(1, int(self.path_length / self.step_size))
        n_steps=5
        direction = 1.0 if rng.rand() > 0.5 else -1.0
        epsilon={var:direction*self.step_size for var in self.start.keys()}
        q = state.copy()
        p = self.draw_momentum(rng)
        q_new = deepcopy(q)
        p_new = deepcopy(p)
        for i in range(n_steps):
            q_new, p_new=self.leapfrog_nocache(q_new, p_new, epsilon)
            #q_new, p_new,cache=self.leapfrog(q_new, p_new, epsilon,cache)
        acceptprob=self.accept(q, q_new, p, p_new)
        if np.isfinite(acceptprob) and (rng.rand() < acceptprob): 
            q = q_new
            p = p_new
            self._accepted += 1
        return q,p,acceptprob

    def acceptance_rate(self,acceptprob):
        return np.mean(acceptprob)

    def leapfrog(self,q, p,epsilon,cache):
        q_new=deepcopy(q)
        p_new=deepcopy(p)
        cache_new=deepcopy(cache)
        grad_q=self.grad(self.X,self.y,q_new,self.hyper)
        eps=1e-8
        for var in self.start.keys():
            cache_new[var] += grad_q[var]**2
            p_new[var]+= (0.5*epsilon[var])*grad_q[var]/ (np.sqrt(cache_new[var]) + eps)
            q_new[var]+= epsilon[var]*p_new[var]/ (np.sqrt(cache_new[var]) + eps)
        grad_q=self.grad(self.X,self.y,q_new,self.hyper)
        for var in self.start.keys():
            cache_new[var] += grad_q[var]**2
            p_new[var]+= (0.5*epsilon[var])*grad_q[var]/ (np.sqrt(cache_new[var]) + eps)
        return q_new, p_new, cache_new

    def leapfrog_nocache(self,q, p,epsilon):
        q_new=deepcopy(q)
        p_new=deepcopy(p)
        grad_q=self.grad(self.X,self.y,q_new,self.hyper)
        for var in self.start.keys():
            p_new[var]+= (0.5*epsilon[var])*grad_q[var]
            q_new[var]+= epsilon[var]*p_new[var]
        grad_q=self.grad(self.X,self.y,q_new,self.hyper)
        for var in self.start.keys():
            p_new[var]+= (0.5*epsilon[var])*grad_q[var]
        return q_new, p_new

    def accept(self,current_q, proposal_q, current_p, proposal_p):
        E_new = self.energy(proposal_q,proposal_p)
        E_current = self.energy(current_q,current_p)
        A = min(1,np.exp(E_current - E_new))
        return A


    def potential_energy(self,p):
        K=0
        for var in self.start.keys():
            #K-=0.5*np.sum(self._inv_mass_matrix[var].reshape(self.start[var].shape)*np.square(p[var]))
            K-=0.5*np.sum(np.square(p[var]))
        return K

    def energy(self, q, p):
        K=-self.potential_energy(p)
        U=-self.logp(self.X,self.y,q,self.hyper)
        return K+U 


    def draw_momentum(self,rng):
        momentum={}
        for var in self.start.keys():
            dim=(np.array(self.start[var])).size
            #rvar=rng.normal(0,self._inv_mass_matrix[var],dim)
            rvar=rng.normal(0,1,dim)
            momentum[var]=rvar.reshape(self.start[var].shape)
        return momentum


    def sample(self,niter=1e4,burnin=1e3,backend=None,rng=None):
        samples=[]
        burnin_samples=[]
        accepted=[]
        if rng==None:
            rng = np.random.RandomState()
        q,p=self.start,self.draw_momentum(rng)
        self.find_reasonable_epsilon(q,rng)
        for i in tqdm(range(int(burnin))):
            q,p,a=self.step(q,p,rng)
            burnin_samples.append(q)
            accepted.append(a)
        #acc_rate=self.acceptance_rate(accepted)
        #print('burnin acceptance rate : {0:.4f}'.format(acc_rate))
        del accepted[:]
        del burnin_samples[:]
        for i in tqdm(range(int(niter))):
            q,p,a=self.step(q,p,rng)
            accepted.append(a)
            samples.append(q)
            if self._verbose and (i%(niter/10)==0):
                print('acceptance rate : {0:.4f}'.format(self.acceptance_rate(accepted)) )
        posterior={var:[] for var in self.start.keys()}
        logp_samples=np.zeros(len(samples))
        i=0
        for s in samples:
            logp_samples[i]=self.logp(self.X,self.y,s,self.hyper)
            i=i+1
            for var in self.start.keys():
                posterior[var].append(s[var].reshape(-1))
        for var in self.start.keys():
            posterior[var]=np.array(posterior[var])
        #else:
        #    posterior=h5py.File(backend,'w')
        #    num_samples=int(niter)
        #    dset = {var:posterior.create_dataset(var, (num_samples,self.start[var].reshape(-1).shape[0]), maxshape=(None,self.start[var].reshape(-1).shape[0]) ) for var in self.start.keys()}
        #    for i in tqdm(range(int(niter)),total=int(niter)):
        #        q,p=self.step(q,p)
        #        for var in self.start.keys():
        #            dset[var][-1,:]=q[var].reshape(-1)
        #    posterior.flush()
        return posterior,logp_samples

    def multicore_sample(self,niter=1e4,burnin=1e3,backend=None,ncores=cpu_count()):
        pool = Pool(processes=ncores)
        rng = [np.random.RandomState(i) for i in range(ncores)]
        results=pool.map(unwrap_self_mcmc, zip([self]*ncores, [int(niter/ncores)]*ncores,[burnin]*ncores,[backend]*ncores,rng))
        posterior={var:np.concatenate([r[0][var] for r in results],axis=0) for var in self.start.keys()}
        logp_samples=np.concatenate([r[1] for r in results],axis=0)
        return posterior,logp_samples

    def compute_mass_matrix(self,samples,alpha=0.9):
        posterior={var:[] for var in self.start.keys()}
        for s in samples:
            for var in self.start.keys():
                posterior[var].append(s[var].reshape(-1))
        for var in self.start.keys():
            posterior[var]=np.array(posterior[var])
            self._mass_matrix[var]=alpha*self._mass_matrix[var]+(1-alpha)*np.var(posterior[var],axis=0)
            self._inv_mass_matrix[var]=1./self._mass_matrix[var]
            
    def find_reasonable_epsilon(self,state,rng):
        q =state.copy()
        p = self.draw_momentum(rng)
        direction = 1.0 if rng.rand() > 0.5 else -1.0
        epsilon={var:direction*self.step_size for var in self.start.keys()}
        cache = {var:np.zeros_like(self.start[var]) for var in self.start.keys()}
        #q_new, p_new,cache = self.leapfrog(q, p, epsilon,cache)
        q_new, p_new = self.leapfrog_nocache(q, p, epsilon)
        acceptprob=self.accept(q, q_new, p, p_new)
        while (0.5 > acceptprob):
            direction*=1.0
            self.step_size*=0.5
            epsilon={var:direction*self.step_size for var in self.start.keys()}
            cache = {var:np.zeros_like(self.start[var]) for var in self.start.keys()}
            #q_new, p_new,cache = self.leapfrog(q, p, epsilon,cache)
            q_new, p_new = self.leapfrog_nocache(q, p, epsilon)
            acceptprob=self.accept(q, q_new, p, p_new)
        print('step_size {0:.4f}, acceptance prob: {1:.2f}, direction : {2:.2f}'.format(self.step_size,acceptprob,direction))