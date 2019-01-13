
import numpy as np
import scipy as sp
import os
from utils import *
from numpy.linalg import inv
from copy import deepcopy
from multiprocessing import Pool,cpu_count
import os 
from hmc import HMC
from tqdm import tqdm, trange
import h5py 

def unwrap_self_sgmcmc(arg, **kwarg):
    return SGLD.sample(*arg, **kwarg)

class SGLD(HMC):

    def step(self,X_batch,y_batch,state,rng):
        q = state.copy()
        q_new = deepcopy(q)
        n_data=np.float(self.y.shape[0])
        epsilon={var:self.step_size/n_data for var in self.start.keys()}
        #n_x,n_y=X_batch.shape
        #Z=np.random.binomial(1,0.5,n_x*n_y).reshape((n_x,n_y))
        #X_batch_dropout=np.multiply(X_batch,Z)
        q_new = self.langevin(q_new, epsilon,X_batch,y_batch,rng)
        return q_new

    def langevin(self,q,epsilon,X_batch,y_batch,rng):
        q_new=deepcopy(q)
        grad_q=self.grad(X_batch,y_batch,q_new,self.hyper)
        n_batch=np.float(y_batch.shape[0])
        n_data=np.float(self.y.shape[0])
        for var in self.start.keys():
            noise_scale = 2.0*epsilon[var]
            sigma = np.sqrt(max(noise_scale, 1e-16)) 
            dim=(np.array(self.start[var])).size
            nu=sigma*rng.normal(0,sigma,dim).reshape(q_new[var].shape)
            q_new[var]+=(n_data/n_batch)*epsilon[var] * grad_q[var]+nu
        return q_new

    def sample(self,niter=1e4,burnin=1e3,batch_size=20,backend=None,rng=None):
        samples=[]
        burnin_samples=[]
        logp_samples=[]
        if rng==None:
            rng = np.random.RandomState(0)
        burnin=int(burnin)
        q=self.start
        n_data=np.float(self.y.shape[0])
        for i in tqdm(range(burnin),total=burnin):
            for X_batch, y_batch in self.iterate_minibatches(self.X, self.y, batch_size):
                q=self.step(X_batch,y_batch,q,rng)
                burnin_samples.append(q)
        del burnin_samples[:]
        for i in tqdm(range(int(niter)),total=int(niter)):
            for batch in self.iterate_minibatches(self.X, self.y, batch_size):
                X_batch, y_batch = batch
                q=self.step(X_batch,y_batch,q,rng)
                samples.append(q)
                logp_samples.append(self.logp(X_batch,y_batch,q,self.hyper))
        posterior={var:[] for var in self.start.keys()}
        for s in samples:
            for var in self.start.keys():
                posterior[var].append(s[var].reshape(-1))
        for var in self.start.keys():
            posterior[var]=np.array(posterior[var])
        return posterior,np.array(logp_samples) 
            
    def iterate_minibatches(self,X, y, batchsize):
        assert X.shape[0] == y.shape[0]
        for start_idx in range(0, X.shape[0] - batchsize + 1, batchsize):
            excerpt = slice(start_idx, start_idx + batchsize)
            yield X[excerpt], y[excerpt]
    
    def multicore_sample(self,niter=1e4,burnin=1e3,batch_size=20,backend=None,ncores=cpu_count()):
        pool = Pool(processes=ncores)
        rng = [np.random.RandomState(i) for i in range(ncores)]
        results=pool.map(unwrap_self_sgmcmc, zip([self]*ncores, [int(niter/ncores)]*ncores,[burnin]*ncores,[batch_size]*ncores,[backend]*ncores,rng))
        posterior={var:np.concatenate([r[0][var] for r in results],axis=0) for var in self.start.keys()}
        logp_samples=np.concatenate([r[1] for r in results],axis=0)
        return posterior,logp_samples
        