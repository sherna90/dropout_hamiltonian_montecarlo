import numpy as np
import scipy as sp
import os
from utils import *
from numpy.linalg import inv
from copy import deepcopy
from multiprocessing import Pool,cpu_count, Process, Queue
import os 
from hamiltonian.hmccpu import HMC
from tqdm import tqdm, trange
import h5py 
import time
import cupy as cp

def unwrap_self_sgmcmc(arg, **kwarg):
    return SGLD.sample(*arg, **kwarg)

def unwrap_self_mean(arg, **kwarg):
    return HMC.sample_mean(*arg, **kwarg)

class SGLD(HMC):

    def step(self,X_batch,y_batch,q_weights,q_bias,rng):
        q_new_weights = deepcopy(q_weights)
        q_new_bias = deepcopy(q_bias)
        n_data=np.float(self.y.shape[0])
        epsilon={var:self.step_size/n_data for var in self.start.keys()}
        #n_x,n_y=X_batch.shape
        #Z=np.random.binomial(1,0.5,n_x*n_y).reshape((n_x,n_y))
        #X_batch_dropout=np.multiply(X_batch,Z)
        q_new = self.langevin(q_new_weights,q_new_bias, epsilon,X_batch,y_batch,rng)
        return q_new

    def langevin(self,q_weights,q_bias,epsilon,X_batch,y_batch,rng):
        q_new_weights = deepcopy(q_weights)
        q_new_bias = deepcopy(q_bias)

        grad_q_weights, grad_q_bias=self.grad(X_batch,y_batch,q_new_weights,q_new_bias,self.hyper)
        
        n_batch=np.float(y_batch.shape[0])
        n_data=np.float(self.y.shape[0])

        noise_scale = 2.0*epsilon['weights']
        sigma = np.sqrt(max(noise_scale, 1e-16)) 
        dim=(np.array(self.start['weights'])).size
        nu=sigma*rng.normal(0,sigma,dim).reshape(q_new_weights.shape)        
        q_new_weights+=(n_data/n_batch)*epsilon['weights'] * grad_q_weights+cp.asarray(nu)

        noise_scale = 2.0*epsilon['bias']
        sigma = np.sqrt(max(noise_scale, 1e-16)) 
        dim=(np.array(self.start['bias'])).size
        nu=sigma*rng.normal(0,sigma,dim).reshape(q_new_bias.shape)
        q_new_bias+=(n_data/n_batch)*epsilon['bias'] * grad_q_bias+cp.asarray(nu)

        return q_new_weights, q_new_bias

        '''
        for var in self.start.keys():
            noise_scale = 2.0*epsilon[var]
            sigma = np.sqrt(max(noise_scale, 1e-16)) 
            dim=(np.array(self.start[var])).size
            nu=sigma*rng.normal(0,sigma,dim).reshape(q_new[var].shape)
            q_new[var]+=(n_data/n_batch)*epsilon[var] * grad_q[var]+nu
        return q_new
        '''

    def sample(self,niter=1e4,burnin=1e3,batchsize=20,backend=None,rng=None):
        q_weights=cp.asarray(self.start['weights'])
        q_bias=cp.asarray(self.start['bias'])

        for i in tqdm(range(int(burnin)),total=int(burnin)):
            for auxiliar in range(len(range(0, self.X.shape[0] - batchsize + 1, batchsize))):
                X_batch, y_batch = SGLD.sample.q.get()
                q_weights, q_bias=self.step(X_batch,y_batch,q_weights,q_bias,rng)

        logp_samples=np.zeros(niter)
        if backend:
            backend_samples=h5py.File(backend)
            posterior={}
            for var in self.start.keys():
                param_shape=self.start[var].shape
                posterior[var]=backend_samples.create_dataset(var,(1,)+param_shape,maxshape=(None,)+param_shape,dtype=np.float32)
            for i in tqdm(range(int(niter)),total=int(niter)):
                for auxiliar in range(len(range(0, self.X.shape[0] - batchsize + 1, batchsize))):
                    X_batch, y_batch = SGLD.sample.q.get()
                    q=self.step(X_batch,y_batch,q,rng)
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
                for auxiliar in range(len(range(0, self.X.shape[0] - batchsize + 1, batchsize))):
                    X_batch, y_batch = SGLD.sample.q.get()
                    q_weights, q_bias=self.step(X_batch,y_batch,q_weights,q_bias,rng)
                    #logp_samples[i] = self.logp(X_batch,y_batch,q,self.hyper)

                    posterior['weights'].append(q_weights.reshape(-1))
                    posterior['bias'].append(q_bias.reshape(-1))

                    #for var in self.start.keys():
                    #    posterior[var].append(q[var].reshape(-1))

            #posterior['weights']=cp.asnumpy(posterior['weights'])
            #posterior['bias']=np.array(posterior['bias'])

            #for var in self.start.keys():
            #    posterior[var]=np.array(posterior[var])
            # **print(np.sum(np.array(posterior['weights']),axis=0))**
            return posterior['weights'], posterior['bias'],logp_samples
            
    def iterate_minibatches(self,X, y, batchsize):
        assert X.shape[0] == y.shape[0]
        for start_idx in range(0, X.shape[0] - batchsize + 1, batchsize):
            excerpt = slice(start_idx, start_idx + batchsize)
            yield X[excerpt], y[excerpt]

    def f1(self, q, batchsize, total):
        for i in range(int(total)):
            #assert self.X.shape[0] == self.y.shape[0]
            for start_idx in range(0, self.X.shape[0] - batchsize + 1, batchsize):
                excerpt = slice(start_idx, start_idx + batchsize)
                q.put((cp.asarray(self.X[excerpt]),cp.asarray(self.y[excerpt])))

    def sample_init(self, q):
        SGLD.sample.q = q
    
    def multicore_sample(self,niter=1e4,burnin=1e3,batch_size=20,backend=None,ncores=cpu_count()):
        if backend:
            multi_backend = [backend+"_%i.h5" %i for i in range(ncores)]
        else:
            multi_backend = [backend]*ncores
    
        rng = [np.random.RandomState(i) for i in range(ncores)]

        ################## QUEUE ##################
        q = Queue(maxsize=ncores)
        
        l = Process(target=SGLD.f1, args=(self, q, batch_size, (int(niter/ncores)*ncores + int(burnin/ncores)*ncores)))

        p = Pool(None, SGLD.sample_init, [self, q])

        l.start()
        ################## QUEUE ##################

        #pool = Pool(processes=ncores)
        results=p.map(unwrap_self_sgmcmc, zip([self]*ncores, [int(niter/ncores)]*ncores,[int(burnin/ncores)]*ncores,[batch_size]*ncores, multi_backend,rng))
        print("VOLVIO...")
        time.sleep(100)

        l.join() # ? #

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
        #pool = Pool(processes=ncores)
        #results= pool.map(unwrap_self_mean, zip([self]*ncores, multi_backend))
        #aux={var:((np.sum([r[var] for r in results],axis=0).reshape(self.start[var].shape))/niter) for var in self.start.keys()}
        #return aux

    def sample_mean(self, filename):
        f=h5py.File(filename)
        aux = {var:np.sum(f[var],axis=0) for var in f.keys()}
        return aux
        