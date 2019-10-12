import numpy as np
from hamiltonian.utils import *
from numpy.linalg import inv,norm
from copy import deepcopy
from tqdm import tqdm, trange
import h5py 
import os 
from multiprocessing import Pool,cpu_count

class hmc:
    def __init__(self,model, start_p, path_length=1.0,step_size=0.1,verbose=True):
        self.start = start_p
        self.step_size = step_size
        self.path_length = path_length
        self.model = model
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
        self.verbose=verbose


    def step(self,state,momentum,rng,**args):
        q = state.copy()
        p = self.draw_momentum(rng)
        q_new = deepcopy(q)
        p_new = deepcopy(p)
        positions, momentums = [deepcopy(q)], [deepcopy(p)]
        path_length=2 * np.random.rand() * self.path_length
        epsilon=self.step_size
        grad_q=self.model.grad(q_new,**args)
        for _ in range(int(path_length/epsilon)):
            for var in self.start.keys():
                p_new[var]-= (0.5*epsilon)*grad_q[var]
                q_new[var]+= epsilon*p_new[var]
            grad_q=self.model.grad(q_new,**args)
            for var in self.start.keys():
                p_new[var]-= (0.5*epsilon)*grad_q[var]
            positions.append(deepcopy(q_new)) 
            momentums.append(deepcopy(p_new)) 
        for var in self.start.keys():
            p_new[var]=-p_new[var]
        acceptprob=self.accept(q, q_new, p, p_new,**args)
        if np.isfinite(acceptprob) and (rng.rand() < acceptprob): 
            q = q_new.copy()
            p = p_new.copy()
        return q,p,positions, momentums


    def accept(self,current_q, proposal_q, current_p, proposal_p,**args):
        E_new = self.energy(proposal_q,proposal_p,**args)
        E_current = self.energy(current_q,current_p,**args)
        A = min(1,np.exp(E_current - E_new))
        return A


    def potential_energy(self,p):
        K=0
        for var in p.keys():
            dim=(np.array(p[var])).size
            #K-=0.5*np.sum(self._inv_mass_matrix[var].reshape(self.start[var].shape)*np.square(p[var]))
            K-=0.5*(dim*np.log(2*np.pi)+np.sum(np.square(p[var])))
        return K

    def energy(self, q, p,**args):
        K=self.potential_energy(p)
        U=self.model.logp(q,**args)
        return K+U 


    def draw_momentum(self,rng):
        momentum={}
        for var in self.start.keys():
            dim=(np.array(self.start[var])).size
            #rvar=rng.normal(0,self._inv_mass_matrix[var],dim)
            rvar=rng.normal(0,1,dim)
            if dim>1:
                momentum[var]=rvar.reshape(self.start[var].shape)
            else:
                momentum[var]=rvar
        return momentum


    def sample(self,niter=1e4,burnin=1e3,rng=None,**args):
        if rng == None:
            rng = np.random.RandomState()
        q,p=self.start,self.draw_momentum(rng)
        #hmcself.find_reasonable_epsilon(q,rng,**args)
        for _ in tqdm(range(int(burnin))):
            q,p,positions,momentums=self.step(q,p,rng,**args)
        logp_samples=np.zeros(int(niter))
        sample_positions, sample_momentums = [], []
        posterior={var:[] for var in self.start.keys()}
        for i in tqdm(range(int(niter))):
            q,p,positions,momentums=self.step(q,p,rng,**args)
            sample_positions.append(positions)
            sample_momentums.append(momentums)
            logp_samples[i]=-1.0*self.model.logp(q,**args)
            for var in self.start.keys():
                posterior[var].append(q[var])
            if self.verbose and (i%(niter/10)==0):
                print('loss: {0:.4f}'.format(logp_samples[i]))
        for var in self.start.keys():
            posterior[var]=np.array(posterior[var])
        return posterior,sample_positions,sample_momentums,logp_samples

            
    def find_reasonable_epsilon(self,state,rng,**args):
        q =state.copy()
        p = self.draw_momentum(rng)
        direction = 1.0 if rng.rand() > 0.5 else -1.0
        epsilon={var:direction*self.step_size for var in self.start.keys()}
        cache = {var:np.zeros_like(self.start[var]) for var in self.start.keys()}
        #q_new, p_new,cache = self.leapfrog(q, p, epsilon,cache)
        q_new, p_new = self.leapfrog(q, p,**args)
        while (0.5 > acceptprob):
            direction*=1.0
            self.step_size*=0.5
            epsilon={var:direction*self.step_size for var in self.start.keys()}
            cache = {var:np.zeros_like(self.start[var]) for var in self.start.keys()}
            q_new, p_new = self.leapfrog(q, p,**args)
        #print
        #print('step_size {0:.4f}, acceptance prob: {1:.2f}, direction : {2:.2f}'.format(self.step_size,acceptprob,direction))

    def backend_mean(self, multi_backend, niter, ncores=cpu_count()):
        aux = []
        for filename in multi_backend:
            f=h5py.File(filename)
            aux.append({var:np.sum(f[var],axis=0) for var in f.keys()})
        mean = {var:((np.sum([r[var] for r in aux],axis=0).reshape(self.start[var].shape))/niter) for var in self.start.keys()}
        return mean
        