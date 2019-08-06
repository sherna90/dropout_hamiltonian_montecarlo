import numpy as np
import scipy as sp
import os
from hamiltonian.utils import *
from numpy.linalg import inv,norm
from copy import deepcopy
import os 
from hamiltonian.cpu.hmc import hmc
from tqdm import tqdm, trange
import h5py 
import time

class sgld(hmc):

    def step(self,momentum,X_batch,y_batch,state,rng):
        p_new=deepcopy(momentum)
        q_new = deepcopy(state)
        n_batch=np.float(y_batch.shape[0])
        epsilon=self.step_size
        grad_q=self.model.grad(X_batch,y_batch,state,self.hyper)
        for var in q_new.keys():
            noise_scale = 2.0*epsilon
            sigma = np.sqrt(max(noise_scale, 1e-16)) 
            dim=(np.array(self.start[var])).size
            nu=rng.normal(0,1,dim).reshape(q_new[var].shape)
            p_new[var] = nu*p_new[var] + (1.0/n_batch)*epsilon * grad_q[var]/norm(grad_q[var])
            q_new[var]+=p_new[var]
        return q_new,p_new


    def sample(self,X_train,y_train,niter=1e4,burnin=1e3,batch_size=20,backend=None):
        rng = np.random.RandomState()
        q=self.start
        momentum={var:np.zeros_like(self.start[var]) for var in self.start.keys()}
        for i in tqdm(range(int(burnin)),total=int(burnin)):
            j=0
            for X_batch, y_batch in self.iterate_minibatches(X_train, y_train, batch_size):
                q,momentum=self.step(momentum,X_batch,y_batch,q,rng)
                if (j%100 == 0):
                    iter_loss=-1.0*self.model.log_likelihood(X_batch,y_batch,q,self.hyper)
                    print('burin minibatch : {0}, loss: {1:.4f}'.format(j,iter_loss))
                j+=1
        logp_samples=np.zeros(int(niter))
        if backend:
            backend_samples=h5py.File(backend)
            posterior={}
            for var in self.start.keys():
                param_shape=self.start[var].shape
                posterior[var]=backend_samples.create_dataset(var,(1,)+param_shape,maxshape=(None,)+param_shape,dtype=np.float32)
            momentum={var:np.zeros_like(self.start[var]) for var in self.start.keys()}
            for i in tqdm(range(int(niter)),total=int(niter)):
                for X_batch, y_batch in self.iterate_minibatches(X_train, y_train, batch_size):
                    q,momentum=self.step(momentum,X_batch,y_batch,q,rng)
                logp_samples[i] = -1.0*self.model.log_likelihood(X_batch,y_batch,q,self.hyper)
                for var in self.start.keys():
                    param_shape=self.start[var].shape
                    posterior[var][-1,:]=q[var]
                    posterior[var].resize((posterior[var].shape[0]+1,)+param_shape)
                backend_samples.flush()
            backend_samples.close()
            return [backend], logp_samples
        else:
            posterior={var:[] for var in self.start.keys()}
            momentum={var:np.zeros_like(self.start[var]) for var in self.start.keys()}
            for i in tqdm(range(int(niter)),total=int(niter)):
                for X_batch, y_batch in self.iterate_minibatches(X_train, y_train, batch_size):
                    q,momentum=self.step(momentum,X_batch,y_batch,q,rng)
                logp_samples[i] = -1.*self.model.log_likelihood(X_batch,y_batch,q,self.hyper)
                for var in self.start.keys():
                        posterior[var].append(q[var].reshape(-1))
            for var in self.start.keys():
                posterior[var]=np.array(posterior[var])
            return posterior,logp_samples
            
    def iterate_minibatches(self,X_train,y_train,batchsize):
        assert X_train.shape[0] == y_train.shape[0]
        for start_idx in range(0, X_train.shape[0] - batchsize + 1, batchsize):
            excerpt = slice(start_idx, start_idx + batchsize)
            yield X_train[excerpt], y_train[excerpt]

    def backend_mean(self, backend, niter):
        backend_samples=h5py.File(backend)
        mean = {var:np.mean(backend_samples[var][:],axis=0) for var in backend_samples.keys()}
        backend_samples.close()
        return mean
        