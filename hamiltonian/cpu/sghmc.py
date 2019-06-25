import numpy as np
import scipy as sp
import os
from hamiltonian.utils import *
from numpy.linalg import inv
from copy import deepcopy
from multiprocessing import Pool,cpu_count
import os 

from hamiltonian.cpu.hmc import hmc

from tqdm import tqdm, trange
import h5py 
import time

class sghmc(hmc):

    def step(self,X_train,y_train,X_batch,y_batch,state,momemtum,rng):
        n_steps=1
        direction = 1.0 if rng.rand() > 0.5 else -1.0
        epsilon={var:direction*self.step_size for var in self.start.keys()}
        q = state.copy()
        p = self.draw_momentum(rng)
        q_new = deepcopy(q)
        p_new = deepcopy(p)
        for i in range(n_steps):
            q_new, p_new = self.leapfrog(q_new,p_new, epsilon,X_batch,y_batch,rng)
        
        acceptprob=self.accept(X_train,y_train,q, q_new, p, p_new)
        if np.isfinite(acceptprob) and (rng.rand() < acceptprob):
            q = q_new
            p = p_new
            self._accepted += 1
        return q,p,acceptprob

    def leapfrog(self,q, p,epsilon,X_batch,y_batch,rng):
        alpha=0.9
        q_new=deepcopy(q)
        p_new=deepcopy(p)
        grad_q=self.grad(X_batch,y_batch,q_new,self.hyper)
        n_data=np.float(y_batch.shape[0])
        for var in self.start.keys():
            noise_scale = 2.0*alpha*epsilon[var]
            sigma = np.sqrt(max(noise_scale, 1e-16)) 
            dim=(np.array(self.start[var])).size
            nu=sigma*rng.normal(0,sigma,dim).reshape(p_new[var].shape)
            p_new[var] = (1.0-alpha) * p_new[var] + (1./n_data)*epsilon[var] * grad_q[var]+nu
            q_new[var]+=p_new[var]
        return q_new, p_new

    def sample(self,X_train,y_train,niter=1e4,burnin=1e3,batch_size=20,backend=None,rng=None):
        accepted=[]
        rng = np.random.RandomState()
        q,p=self.start,self.draw_momentum(rng)
        #self.find_reasonable_epsilon(X_train,y_train,q,rng)
        for i in tqdm(range(int(burnin)),total=int(burnin)):
            for X_batch, y_batch in self.iterate_minibatches(X_train, y_train, batch_size):
                q,p,a=self.step(X_train,y_train,X_batch,y_batch,q,p,rng)
                accepted.append(a)
        if self._verbose:
            print('burn-in acceptance rate : {0:.4f}'.format(self.acceptance_rate(accepted)))  
        
        del accepted[:]
        logp_samples=np.zeros(int(niter))
        if backend:
            backend_samples=h5py.File(backend)
            posterior={}
            for var in self.start.keys():
                param_shape=self.start[var].shape
                posterior[var]=backend_samples.create_dataset(var,(1,)+param_shape,maxshape=(None,)+param_shape,dtype=np.float32)
            for i in tqdm(range(int(niter)),total=int(niter)):
                for X_batch, y_batch in self.iterate_minibatches(X_train, y_train, batch_size):
                    q,p,a=self.step(X_batch,y_batch,q,p,rng)
                    logp_samples[i] = self.logp(X_batch,y_batch,q,self.hyper)
                    for var in self.start.keys():
                        param_shape=self.start[var].shape
                        posterior[var].resize((posterior[var].shape[0]+1,)+param_shape)
                        posterior[var][-1,:]=q[var]
                    backend_samples.flush()
            backend_samples.close()
            return 1, logp_samples
        else:
            posterior={var:[] for var in self.start.keys()}
            for i in tqdm(range(int(niter)),total=int(niter)):
                for X_batch, y_batch in self.iterate_minibatches(X_train, y_train, batch_size):
                    q,p,a=self.step(X_train,y_train,X_batch,y_batch,q,p,rng)
                    accepted.append(a)
                    logp_samples[i] = self.logp(X_batch,y_batch,q,self.hyper)
                    for var in self.start.keys():
                        posterior[var].append(q[var].reshape(-1))
                if self._verbose:
                    print('acceptance rate : {0:.4f}'.format(self.acceptance_rate(accepted)))        

            for var in self.start.keys():
                posterior[var]=np.array(posterior[var])
            return posterior,logp_samples
            
    def iterate_minibatches(self,X_train,y_train, batchsize):
        assert X_train.shape[0] == y_train.shape[0]
        for start_idx in range(0, X_train.shape[0] - batchsize + 1, batchsize):
            excerpt = slice(start_idx, start_idx + batchsize)
            yield X_train[excerpt], y_train[excerpt]
    
    def backend_mean(self, multi_backend, niter):
        aux = []
        for filename in multi_backend:
            f=h5py.File(filename)
            aux.append({var:np.sum(f[var],axis=0) for var in f.keys()})
        mean = {var:((np.sum([r[var] for r in aux],axis=0).reshape(self.start[var].shape))/niter) for var in self.start.keys()}
        return mean
        