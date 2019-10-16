import numpy as np
import scipy as sp
import os
from hamiltonian.utils import *
from numpy.linalg import inv,norm
from copy import deepcopy
from multiprocessing import Pool,cpu_count
import os 

from hamiltonian.inference.cpu.hmc import hmc

from tqdm import tqdm, trange
import h5py 
import time

class sghmc(hmc):
    

    def iterate_minibatches(self, X, y, batchsize):
        assert X.shape[0] == y.shape[0]
        for start_idx in range(0, X.shape[0] - batchsize + 1, batchsize):
            excerpt = slice(start_idx, start_idx + batchsize)
            yield X[excerpt], y[excerpt]

    

    def step(self,state,momentum,rng,**args):
        q = state.copy()
        p = self.draw_momentum(rng)
        q_new = deepcopy(q)
        p_new = deepcopy(p)
        epsilon=self.step_size
        path_length=np.ceil(2*np.random.rand()*self.path_length/epsilon)
        grad_q=self.model.grad(q,**args)
        # SG-HMC leapfrog step 
        for _ in np.arange(path_length-1):
            for var in self.start.keys():
                dim=(np.array(q_new[var])).size
                rvar=rng.normal(0,2*epsilon,dim).reshape(q[var].shape)
                q_new[var]+= epsilon*p_new[var]
                grad_q=self.model.grad(q_new,**args)
                p_new[var] = (1-epsilon)*p_new[var] + epsilon*grad_q[var]+rvar 
        acceptprob = self.accept(q, q_new, p, p_new,**args)
        if np.isfinite(acceptprob) and (np.random.rand() < acceptprob): 
            q = q_new.copy()
            p = p_new.copy()
        return q,p,acceptprob

    def sample(self,epochs=1,burnin=1,batch_size=1,rng=None,**args):
        if rng == None:
            rng = np.random.RandomState()
        X=args['X_train']
        y=args['y_train']
        if 'verbose' in args:
            verbose=args['verbose']
        else:
            verbose=None
        epochs=int(epochs)
        q,p=self.start,self.draw_momentum(rng)
        for _ in tqdm(range(int(burnin))):
            for X_batch, y_batch in self.iterate_minibatches(X, y, batch_size):
                kwargs={'X_train':X_batch,'y_train':y_batch,'verbose':verbose}
                q,p,p_accept=self.step(q,p,rng,**kwargs)
        logp_samples=np.zeros(epochs)
        posterior={var:[] for var in self.start.keys()}
        for i in tqdm(range(epochs)):
            for X_batch, y_batch in self.iterate_minibatches(X, y, batch_size):
                kwargs={'X_train':X_batch,'y_train':y_batch,'verbose':verbose}
                q,p,p_accept=self.step(q,p,rng,**kwargs)
            logp_samples[i]=self.model.logp(q,**args)
            for var in self.start.keys():
                posterior[var].append(q[var])
            if self.verbose and (i%(epochs/10)==0):
                print('loss: {0:.4f}'.format(logp_samples[i]))
        for var in self.start.keys():
            posterior[var]=np.array(posterior[var])
        return posterior,logp_samples