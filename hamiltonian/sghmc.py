
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
    return SGHMC.sample(*arg, **kwarg)

class SGHMC(HMC):

    def step(self,X_batch,y_batch,state,momemtum,rng):
        #n_steps = max(1, int(self.path_length / self.step_size))
        n_steps=5
        direction = 1.0 if rng.rand() > 0.5 else -1.0
        epsilon={var:direction*self.step_size for var in self.start.keys()}
        q = state.copy()
        p = self.draw_momentum(rng)
        q_new = deepcopy(q)
        p_new = deepcopy(p)
        for i in range(n_steps):
            q_new, p_new = self.leapfrog(q_new,p_new, epsilon,X_batch,y_batch,rng)
        acceptprob=self.accept(q, q_new, p, p_new)
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

    def sample(self,niter=1e4,burnin=1e3,batch_size=20,backend=None,rng=None):
        samples=[]
        burnin_samples=[]
        accepted=[]
        logp_samples=[]
        if rng==None:
            rng = np.random.RandomState(0)
        burnin=int(burnin)
        q,p=self.start,self.draw_momentum(rng)
        self.find_reasonable_epsilon(q,rng)
        for i in tqdm(range(burnin),total=burnin):
            for batch in self.iterate_minibatches(self.X, self.y, batch_size):
                X_batch, y_batch = batch
                q,p,a=self.step(X_batch,y_batch,q,p,rng)
                burnin_samples.append(q)
                accepted.append(a)
        del accepted[:]
        del burnin_samples[:]
        for i in tqdm(range(int(niter)),total=int(niter)):
            for batch in self.iterate_minibatches(self.X, self.y, batch_size):
                X_batch, y_batch = batch
                q,p,a=self.step(X_batch,y_batch,q,p,rng)
                samples.append(q)
                accepted.append(a)
                logp_samples.append(self.logp(X_batch,y_batch,q,self.hyper))
            if self._verbose and (i%(niter/10)==0):
                print('acceptance rate : {0:.4f}'.format(self.acceptance_rate(accepted)))        
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
        