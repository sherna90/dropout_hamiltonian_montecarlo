import cupy as cp
import scipy as sp
import os
from hamiltonian.utils import *
from cupy.linalg import inv
from copy import deepcopy
from multiprocessing import Pool,cpu_count
from tqdm import tqdm, trange
import h5py 
import os 
import math 
import time
import os

class hmc:

    def __init__(self,model, start_p, path_length=1.0,step_size=0.1,verbose=True):
        self.start={var:cp.asarray(start_p[var]) for var in start_p.keys()}
        self.step_size = step_size
        self.path_length = path_length
        self.model = model
        self._mass_matrix={}
        self._inv_mass_matrix={}
        self._mass_matrix={}
        self._inv_mass_matrix={}
        for var in self.start.keys():
            dim=(cp.array(self.start[var])).size
            if dim==1:
                self._mass_matrix[var]=cp.array(1.0)
                self._inv_mass_matrix[var]=cp.array(1.0)
            else:
                self._mass_matrix[var]=cp.ones(dim)
                self._inv_mass_matrix[var]=cp.ones(dim)
        self.verbose=verbose
        self.mu = cp.log(10 * self.step_size)  # proposals are biased upwards to stay away from 0.
        self.target_accept = 0.65
        self.gamma = 0.05
        self.t0 = 10.0
        self.t = 0
        self.kappa = 0.75
        self.error_sum = 0.0
        self.log_averaged_step = 0.0

    def step(self,state,momentum,rng,**args):
        q = state.copy()
        p = self.draw_momentum(rng)
        q_new = deepcopy(q)
        p_new = deepcopy(p)
        positions, momentums = [cp.asnumpy(q)], [cp.asnumpy(p)]
        epsilon=self.step_size
        path_length=cp.ceil(2*cp.random.rand()*self.path_length/epsilon)
        grad_q=self.model.grad(q,**args)
        # half step
        for var in self.start.keys():
            p_new[var]-= (0.5*epsilon)*grad_q[var]
        # leapfrog step 
        for _ in cp.arange(path_length-1):
            for var in self.start.keys():
                q_new[var]=q_new[var]+epsilon*p_new[var]
                grad_q=self.model.grad(q_new,**args)
                p_new[var] =p_new[var] - epsilon*grad_q[var]
            positions.append(deepcopy(q_new)) 
            momentums.append(deepcopy(p_new)) 
        # half step
        for var in self.start.keys():
            q_new[var]+= epsilon*p_new[var]
            grad_q=self.model.grad(q_new,**args)
            p_new[var]=-(0.5*epsilon)*grad_q[var]
        # negate momentum
        for var in self.start.keys():
            p_new[var]=-p_new[var]
        positions.append(deepcopy(q_new)) 
        momentums.append(deepcopy(p_new)) 
        acceptprob = self.accept(q, q_new, p, p_new,**args)
        if cp.isfinite(acceptprob) and (cp.random.rand() < acceptprob): 
            q = q_new.copy()
            p = p_new.copy()
        return q,p,positions, momentums,acceptprob


    def accept(self,current_q, proposal_q, current_p, proposal_p,**args):
        E_new = (self.model.negative_log_posterior(proposal_q,**args)+self.potential_energy(proposal_p))
        E_current = (self.model.negative_log_posterior(current_q,**args)+self.potential_energy(current_p))
        A = min(1,cp.exp(E_current-E_new))
        return A


    def potential_energy(self,p):
        K=0
        for var in p.keys():
            dim=(cp.array(p[var])).size
            K-=0.5*(dim*cp.log(2*cp.pi)+cp.sum(cp.square(p[var])))
        return K


    def draw_momentum(self,rng):
        momentum={}
        for var in self.start.keys():
            dim=(cp.array(self.start[var])).size
            momentum[var]=rng.normal(0,1,size=self.start[var].shape)
        return momentum


    def sample(self,niter=1e4,burnin=1e3,rng=None,**args):
        if rng == None:
            rng = cp.random.RandomState()
        q,p=self.start,self.draw_momentum(rng)
        for _ in tqdm(range(int(burnin))):
            q,p,positions,momentums,p_accept=self.step(q,p,rng,**args)
            #self.step_size,_=step_size_tuning.update(p_accept)
        loss=np.zeros(int(niter))
        sample_positions, sample_momentums = [], []
        posterior={var:[] for var in self.start.keys()}
        for i in tqdm(range(int(niter))):
            q,p,positions,momentums,_=self.step(q,p,rng,**args)
            sample_positions.append(positions)
            sample_momentums.append(momentums)
            loss[i]=cp.asnumpy(self.model.negative_log_posterior(q,**args))
            for var in self.start.keys():
                posterior[var].append(q[var])
            if self.verbose is not None and (i%(niter/10)==0):
                print('loss: {0:.4f}'.format(loss[i]))
        for var in self.start.keys():
            temp=cp.asarray(posterior[var],dtype=float)
            posterior[var]=cp.asnumpy(temp)
        return posterior,loss,sample_positions,sample_momentums

            
    def find_reasonable_epsilon(self,p_accept,**args):
        self.t += 1
        g=self.target_accept - p_accept
        self.error_sum += self.target_accept - p_accept
        #self.error_sum  = (1.0 - 1.0/(self.t + self.t0)) * self.error_sum + g/(self.t + self.t0)
        log_step = self.mu - self.prox_center - (self.t ** 0.5) / self.gamma * self.error_sum
        eta = self.t **(-self.kappa)
        self.log_averaged_step = eta * log_step + (1 - eta) * self.log_averaged_step
        return cp.exp(log_step), cp.exp(self.log_averaged_step)

    def backend_mean(self, multi_backend, niter, ncores=cpu_count()):
        aux = []
        for filename in multi_backend:
            f=h5py.File(filename)
            aux.append({var:cp.sum(f[var],axis=0) for var in f.keys()})
        mean = {var:((cp.sum([r[var] for r in aux],axis=0).reshape(self.start[var].shape))/niter) for var in self.start.keys()}
        return mean
        
