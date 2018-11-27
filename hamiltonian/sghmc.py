
import numpy as np
import scipy as sp
import os
from utils import *
from numpy.linalg import inv
from copy import deepcopy
from multiprocessing import Pool,current_process
import os 
from hmc import HMC
from tqdm import tqdm, trange
import h5py 

def unwrap_self_sgmcmc(arg, **kwarg):
    return SGHMC.sample(*arg, **kwarg)

class SGHMC(HMC):

    def step(self,X_batch,y_batch,state,momemtum,rng):
        path_length = rng.rand() * self.path_length
        n_steps = max(1, int(path_length / max(self.step_size.values())))
        direction = 1.0 if rng.rand() > 0.5 else -1.0
        epsilon={var:direction*self.step_size[var] for var in self.start.keys()}
        q = deepcopy(state)
        p = self.draw_momentum(rng)
        q_new=deepcopy(q)
        p_new=deepcopy(p)
        grad_q=self.grad(X_batch,y_batch,q,self.hyper)
        for var in self.start.keys():
            p_new[var]-= (0.5*epsilon[var])*grad_q[var]
            q_new[var]+=epsilon[var]*self._inv_mass_matrix[var].reshape(self.start[var].shape)*p_new[var]
        for i in range(n_steps-1):
            q_new, p_new = self.leapfrog(q_new, p_new, epsilon,X_batch,y_batch)
        grad_q=self.grad(self.X,self.y,q_new,self.hyper)
        for var in self.start.keys():
           p_new[var]-= (0.5*epsilon[var])*grad_q[var]
        if self.accept(q, q_new, p, p_new):
            q = q_new
            p = p_new
            self._accepted += 1
        return q,p

    def leapfrog(self,q, p,epsilon,X_batch,y_batch):
        grad_q=self.grad(X_batch,y_batch,q,self.hyper)
        for var in self.start.keys():
            p[var]-= (0.5*epsilon[var])*grad_q[var]
            q[var]+=epsilon[var]*self._inv_mass_matrix[var].reshape(self.start[var].shape)*p[var]
        return q, p

    def sample(self,niter=1e4,burnin=1e3,batch_size=20,backend=None,rng=None):
        if rng==None:
            rng = np.random.RandomState(0)
        burnin=int(burnin)
        #proc=current_process()
        samples=[]
        q,p=self.start,self.draw_momentum(rng)
        for i in tqdm(range(burnin),total=burnin):
            for batch in self.iterate_minibatches(self.X, self.y, batch_size):
                X_batch, y_batch = batch
                (q,p)=self.step(X_batch,y_batch,q,p,rng)
        self._accepted = 0
        if backend is None:
            for i in tqdm(range(int(niter)),total=int(niter)):
                for batch in self.iterate_minibatches(self.X, self.y, batch_size):
                    X_batch, y_batch = batch
                    (q,p)=self.step(X_batch,y_batch,q,p,rng)
                    samples.append(q)
            posterior={var:[] for var in self.start.keys()}
            for s in samples:
                for var in self.start.keys():
                    posterior[var].append(s[var].reshape(-1))
            for var in self.start.keys():
                posterior[var]=np.array(posterior[var])
        else:
            posterior=h5py.File(backend,'w')
            num_samples=int(niter*self.X.shape[0]/batch_size)
            dset = {var:posterior.create_dataset(var, (num_samples,self.start[var].reshape(-1).shape[0]), maxshape=(None,self.start[var].reshape(-1).shape[0]) ) for var in self.start.keys()}
            for i in tqdm(range(int(niter)),total=int(niter)):
                for batch in self.iterate_minibatches(self.X, self.y, batch_size):
                    X_batch, y_batch = batch
                    (q,p)=self.step(X_batch,y_batch,q,p)
                    for var in self.start.keys():
                        dset[var][-1,:]=q[var].reshape(-1)       
                    posterior.flush()
        return posterior 
            
    def iterate_minibatches(self,X, y, batchsize):
        assert X.shape[0] == y.shape[0]
        for start_idx in range(0, X.shape[0] - batchsize + 1, batchsize):
            excerpt = slice(start_idx, start_idx + batchsize)
            yield X[excerpt], y[excerpt]
    
    def multicore_sample(self,niter=1e4,burnin=1e3,batch_size=20,backend=None,ncores=2):
        pool = Pool(processes=ncores)
        rng = [np.random.RandomState(i) for i in range(ncores)]
        results=pool.map(unwrap_self_sgmcmc, zip([self]*ncores, [int(niter/ncores)]*ncores,[burnin]*ncores,[batch_size]*ncores,[backend]*ncores,rng))
        posterior={var:np.concatenate([r[var] for r in results],axis=0) for var in self.start.keys()}
        return posterior