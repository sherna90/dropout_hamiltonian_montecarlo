import cupy as cp
import numpy as np
import scipy as sp
import os
from hamiltonian.utils import *
from numpy.linalg import inv
from copy import deepcopy
import os 
from hamiltonian.cpu.hmc import hmc
from tqdm import tqdm, trange
import h5py 

class sgld(hmc):

    def step(self,y_train,X_batch,y_batch,state,rng):
        q_new = deepcopy(state)
        n_data=cp.float(y_train.shape[0])
        epsilon={var:self.step_size/n_data for var in self.start.keys()}
        q_new = self.langevin(y_train,q_new, epsilon,X_batch,y_batch,rng)
        return q_new

    def langevin(self,y_train,q,epsilon,X_batch,y_batch,rng):
        q_new=deepcopy(q)
        grad_q=self.grad(X_batch,y_batch,q_new,self.hyper)
        n_batch=cp.float(y_batch.shape[0])
        n_data=cp.float(y_train.shape[0])
        for var in self.start.keys():
            noise_scale = 2.0*epsilon[var]
            sigma = cp.sqrt(max(noise_scale, 1e-16)) 
            dim=(cp.array(self.start[var])).size
            nu=sigma*rng.normal(0,sigma,dim).reshape(q_new[var].shape)
            q_new[var]+=(n_data/n_batch)*1e-8 * grad_q[var]+nu
        return q_new

    def sample(self,X_train,y_train,niter=1e4,burnin=1e3,batch_size=20,backend=None):
        rng = cp.random.RandomState()
        q={var:cp.asarray(self.start[var]) for var in self.start.keys()}
        for i in tqdm(range(int(burnin)),total=int(burnin)):
            for X_batch, y_batch in self.iterate_minibatches(X_train, y_train, batch_size):
                q=self.step(y_train,X_batch,y_batch,q,rng)

        logp_samples=cp.zeros(int(niter))
        if backend:
            backend_samples=h5py.File(backend)
            posterior={}
            for var in self.start.keys():
                param_shape=self.start[var].shape
                posterior[var]=backend_samples.create_dataset(var,(1,)+param_shape,maxshape=(None,)+param_shape,dtype=cp.float32)
            for i in tqdm(range(int(niter)),total=int(niter)):
                for X_batch, y_batch in self.iterate_minibatches(X_train, y_train, batch_size):
                    q=self.step(y_train,X_batch,y_batch,q,rng)
                    #logp_samples[i] = self.logp(X_batch,y_batch,q,self.hyper)
                    for var in self.start.keys():
                        param_shape=self.start[var].shape
                        posterior[var].resize((posterior[var].shape[0]+1,)+param_shape)
                        posterior[var][-1,:]=cp.asnumpy(q[var])
                    backend_samples.flush()
            backend_samples.close()
            return 1, logp_samples
        else:
            posterior={var:[] for var in self.start.keys()}
            for i in tqdm(range(int(niter)),total=int(niter)):
                for X_batch, y_batch in self.iterate_minibatches(X_train, y_train, batch_size):
                    q=self.step(y_train,X_batch,y_batch,q,rng)
                    #logp_samples[i] = self.logp(X_batch,y_batch,q,self.hyper)
                    for var in self.start.keys():
                        posterior[var].append(cp.asnumpy(q[var].reshape(-1)))
            for var in self.start.keys():
                posterior[var]=np.array(posterior[var])
            return posterior,logp_samples
    
    def iterate_minibatches(self, X, y, batchsize):
        assert X.shape[0] == y.shape[0]
        for start_idx in range(0, X.shape[0] - batchsize + 1, batchsize):
            excerpt = slice(start_idx, start_idx + batchsize)
            yield cp.asarray(X[excerpt]),cp.asarray(y[excerpt])

    def backend_mean(self, multi_backend, niter):
        aux = []
        for filename in multi_backend:
            f=h5py.File(filename)
            aux.append({var:cp.sum(f[var],axis=0) for var in f.keys()})
        mean = {var:((cp.sum([r[var] for r in aux],axis=0).reshape(self.start[var].shape))/niter) for var in self.start.keys()}
        return mean
        