import numpy as np
import scipy as sp
import os
from hamiltonian.utils import *
from numpy.linalg import inv
from copy import deepcopy
from multiprocessing import Pool,cpu_count, Process, Queue
import os 
from hamiltonian.cpu.sgld import sgld

from tqdm import tqdm, trange
import h5py 
import time

def unwrap_self_sgmcmc(arg, **kwarg):
    return sgld_multicore.sample(*arg, **kwarg)

class sgld_multicore(sgld):


    def sample(self,niter=1e4,burnin=1e3,batchsize=20,backend=None,rng=None):
        par=self.start
        for i in tqdm(range(int(burnin)),total=int(burnin)):
            for auxiliar in range(len(range(0, self.X.shape[0] - batchsize + 1, batchsize))):
                X_batch, y_batch = sgld_multicore.sample.queue.get()
                par=self.step(X_batch,y_batch,par,rng)
        logp_samples=np.zeros(niter)
        if backend:
            backend_samples=h5py.File(backend)
            posterior={}
            for var in self.start.keys():
                param_shape=self.start[var].shape
                posterior[var]=backend_samples.create_dataset(var,(1,)+param_shape,maxshape=(None,)+param_shape,dtype=np.float32)
            for i in tqdm(range(int(niter)),total=int(niter)):
                for auxiliar in range(len(range(0, self.X.shape[0] - batchsize + 1, batchsize))):
                    X_batch, y_batch = sgld_multicore.sample.queue.get()
                    par=self.step(X_batch,y_batch,par,rng)
                    logp_samples[i] = self.logp(X_batch,y_batch,par,self.hyper)
                    for var in self.start.keys():
                        param_shape=self.start[var].shape
                        posterior[var].resize((posterior[var].shape[0]+1,)+param_shape)
                        posterior[var][-1,:]=par[var]
                    backend_samples.flush()
            backend_samples.close()
            return 1, logp_samples
        else:
            posterior={var:[] for var in self.start.keys()}
            for i in tqdm(range(int(niter)),total=int(niter)):
                for auxiliar in range(len(range(0, self.X.shape[0] - batchsize + 1, batchsize))):
                    X_batch, y_batch = sgld_multicore.sample.queue.get()
                    par=self.step(X_batch,y_batch,par,rng)
                    logp_samples[i] = self.logp(X_batch,y_batch,par,self.hyper)
                    for var in self.start.keys():
                        posterior[var].append(par[var].reshape(-1))
            for var in self.start.keys():
                posterior[var]=np.array(posterior[var])
                
            return posterior,logp_samples

    def iterate_minibatches(self, q, batchsize, total):
        for i in range(int(total)):
            #assert self.X.shape[0] == self.y.shape[0]
            for start_idx in range(0, self.X.shape[0] - batchsize + 1, batchsize):
                excerpt = slice(start_idx, start_idx + batchsize)
                q.put((self.X[excerpt], self.y[excerpt]))

    def sample_init(self, _queue):
        sgld_multicore.sample.queue = _queue
    
    def multicore_sample(self,niter=1e4,burnin=1e3,batch_size=20,backend=None,ncores=cpu_count()):
        if backend:
            multi_backend = [backend+"_%i.h5" %i for i in range(ncores)]
        else:
            multi_backend = [backend]*ncores    
        rng = [np.random.RandomState(i) for i in range(ncores)]
        queue = Queue(maxsize=ncores)      
        l = Process(target=sgld_multicore.iterate_minibatches, args=(self, queue, batch_size, (int(niter/ncores)*ncores + int(burnin/ncores)*ncores)))
        p = Pool(None, sgld_multicore.sample_init, [self, queue])
        l.start()
        results=p.map(unwrap_self_sgmcmc, zip([self]*ncores, [int(niter/ncores)]*ncores,[int(burnin/ncores)]*ncores,[batch_size]*ncores, multi_backend,rng))
        l.join() 
        if not backend:
            posterior={var:np.concatenate([r[0][var] for r in results],axis=0) for var in self.start.keys()}
            logp_samples=np.concatenate([r[1] for r in results],axis=0)
            return posterior,logp_samples
        else:
            logp_samples=np.concatenate([r[1] for r in results],axis=0)
            return multi_backend,logp_samples

    def multicore_mean(self, multi_backend, niter, ncores=cpu_count()):
        aux = []
        for filename in multi_backend:
            f=h5py.File(filename)
            aux.append({var:np.sum(f[var],axis=0) for var in f.keys()})
        mean = {var:((np.sum([r[var] for r in aux],axis=0).reshape(self.start[var].shape))/niter) for var in self.start.keys()}
        return mean