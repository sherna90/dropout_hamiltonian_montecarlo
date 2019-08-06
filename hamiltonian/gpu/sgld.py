import cupy as cp
import numpy as np
import scipy as sp
import os
from hamiltonian.utils import *
from numpy.linalg import inv
from copy import deepcopy
import os 
from hamiltonian.gpu.hmc import hmc
from tqdm import tqdm, trange
import h5py 

from cupy.linalg import norm

class sgld(hmc):

    def step(self,momentum,X_batch,y_batch,state,rng):
        p_new=deepcopy(momentum)
        q_new = deepcopy(state)
        epsilon=self.step_size
        grad_q=self.model.grad(X_batch,y_batch,state,self.hyper)
        n_batch=cp.float(y_batch.shape[0])
        gamma=0.9
        for var in q_new.keys():
            noise_scale = 2.0*epsilon
            dim=q_new[var].shape
            if len(dim)==1:
                nu=cp.random.randn(dim[0],dtype=cp.float32)
            else:
                nu=cp.random.randn(dim[0],dim[1],dtype=cp.float32)
            #p_new[var] = gamma*p_new[var] + (1.0/n_batch)*epsilon * grad_q[var]/norm(grad_q[var])
            p_new[var] = nu*p_new[var] + (1.0/n_batch)*epsilon * grad_q[var]/norm(grad_q[var])
            q_new[var]+=p_new[var]
        return q_new,p_new

    def sample(self,X_train,y_train,epochs=1e4,burnin=1e3,batch_size=20,backend=None):
        rng = cp.random.RandomState()
        q={var:cp.asarray(self.start[var]) for var in self.start.keys()}
        for i in tqdm(range(int(burnin)),total=int(burnin)):
            j=0
            momentum={var:cp.zeros_like(cp.asarray(self.start[var])) for var in self.start.keys()}
            for X_batch, y_batch in self.iterate_minibatches(X_train, y_train, batch_size):
                q,momentum=self.step(momentum,X_batch,y_batch,q,rng)
                if (j%100 == 0):
                    iter_loss=-1.0*cp.asnumpy(self.model.log_likelihood(X_batch,y_batch,q,self.hyper))
                    print('minibatch : {0}, loss: {1:.4f}'.format(j,iter_loss))
                j+=1
        loss_val=np.zeros(int(epochs))
        if backend:
            backend_samples=h5py.File(backend)
            posterior={}
            for var in self.start.keys():
                param_shape=self.start[var].shape
                posterior[var]=backend_samples.create_dataset(var,(1,)+param_shape,maxshape=(None,)+param_shape,dtype=cp.float32)
            for i in tqdm(range(int(epochs)),total=int(epochs)):
                momentum={var:cp.zeros_like(cp.asarray(self.start[var])) for var in self.start.keys()}
                for X_batch, y_batch in self.iterate_minibatches(X_train, y_train, batch_size):
                    q,momentum=self.step(momentum,X_batch,y_batch,q,rng)
                for var in self.start.keys():
                    param_shape=self.start[var].shape
                    posterior[var][-1,:]=cp.asnumpy(q[var])
                    posterior[var].resize((posterior[var].shape[0]+1,)+param_shape)
                backend_samples.flush()
                loss_val[i] = -1.0*cp.asnumpy(self.model.log_likelihood(X_batch,y_batch,q,self.hyper))
                if (i % (epochs/10)==0):
                    print('loss: {0:.4f}'.format(loss_val[i]))
            backend_samples.close()
            return backend_samples, loss_val
        else:
            posterior={var:[] for var in self.start.keys()}
            for i in tqdm(range(int(epochs)),total=int(epochs)):
                for X_batch, y_batch in self.iterate_minibatches(X_train, y_train, batch_size):
                    q=self.step(y_train,X_batch,y_batch,q,rng)
                for var in self.start.keys():
                    posterior[var].append(cp.asnumpy(q[var].reshape(-1)))
                loss_val[i] = -1.0*cp.asnumpy(self.model.log_likelihood(X_batch,y_batch,q,self.hyper))
                if (i % (epochs/10)==0):
                    print('loss: {0:.4f}'.format(loss_val[i]))
            for var in self.start.keys():
                posterior[var]=np.array(posterior[var])
            return posterior,loss_val 
    
    def iterate_minibatches(self, X, y, batchsize):
        assert X.shape[0] == y.shape[0]
        for start_idx in range(0, X.shape[0] - batchsize + 1, batchsize):
            excerpt = slice(start_idx, start_idx + batchsize)
            yield cp.asarray(X[excerpt]),cp.asarray(y[excerpt])

    def backend_mean(self, backend, epochs):
        backend_samples=h5py.File(backend)
        mean = {var:cp.mean(cp.asarray(backend_samples[var][:]),axis=0) for var in backend_samples.keys()}
        backend_samples.close()
        return mean
        