import numpy as np
import scipy as sp
import os
from utils import *
from numpy.linalg import inv
from copy import deepcopy
from multiprocessing import Pool,cpu_count, Process, Queue
import os 
from hamiltonian.cpu.hmc import hmc
from tqdm import tqdm, trange
import h5py 
import time

def unwrap_self_sgmcmc(arg, **kwarg):
    return SGLD.sample(*arg, **kwarg)

def unwrap_self_mean(arg, **kwarg):
    return hmc.sample_mean(*arg, **kwarg)

class SGLD(hmc):

    def step(self,X_batch,y_batch,state,rng):
        q_new = deepcopy(state)
        n_data=np.float(self.y)
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
        n_data=np.float(self.y)
        for var in self.start.keys():
            noise_scale = 2.0*epsilon[var]
            sigma = np.sqrt(max(noise_scale, 1e-16)) 
            dim=(np.array(self.start[var])).size
            nu=sigma*rng.normal(0,sigma,dim).reshape(q_new[var].shape)
            q_new[var]+=(n_data/n_batch)*epsilon[var] * grad_q[var]+nu
        return q_new

    def sample(self,niter=1e4,burnin=1e3,batchsize=20,backend=None,rng=None):
        q=self.start
        for i in tqdm(range(int(burnin)),total=int(burnin)):
            for auxiliar in range(len(range(0, self.X - batchsize + 1, batchsize))):
                X_batch, y_batch = SGLD.sample.q.get()
                q=self.step(X_batch,y_batch,q,rng)
        
        logp_samples=np.zeros(niter)
        if backend:
            backend_samples=h5py.File(backend)
            posterior={}
            for var in self.start.keys():
                param_shape=self.start[var].shape
                posterior[var]=backend_samples.create_dataset(var,(1,)+param_shape,maxshape=(None,)+param_shape,dtype=np.float32)
            for i in tqdm(range(int(niter)),total=int(niter)):
                for auxiliar in range(len(range(0, self.X - batchsize + 1, batchsize))):
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
                for auxiliar in range(len(range(0, self.X - batchsize + 1, batchsize))):
                    X_batch, y_batch = SGLD.sample.q.get()
                    q=self.step(X_batch,y_batch,q,rng)
                    logp_samples[i] = self.logp(X_batch,y_batch,q,self.hyper)
                    for var in self.start.keys():
                        posterior[var].append(q[var].reshape(-1))
            for var in self.start.keys():
                posterior[var]=np.array(posterior[var])
            return posterior,logp_samples
            
    def iterate_minibatches(self,X, y, batchsize):
        assert X.shape[0] == y.shape[0]
        for start_idx in range(0, X.shape[0] - batchsize + 1, batchsize):
            excerpt = slice(start_idx, start_idx + batchsize)
            yield X[excerpt], y[excerpt]

    def f1(self, q, batchsize, total):
        data_path = '../data/'
        plants_train=h5py.File(data_path+'train_features_labels.h5','r')
        X_train=plants_train['train_features']
        y_train=plants_train['train_labels']

        aaa = len(range(0, self.X - batchsize + 1, batchsize)) * int(total)
        cont = 1

        for i in range(int(total)):
            #assert self.X == self.y
            for start_idx in range(0, self.X - batchsize + 1, batchsize):
                print("{} de {}".format(cont, aaa))
                cont += 1
                excerpt = slice(start_idx, start_idx + batchsize)
                q.put((X_train[excerpt], y_train[excerpt]))
        plants_train.close()

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

        #time.sleep(100)
        ################## QUEUE ##################

        #pool = Pool(processes=ncores)
        results=p.map(unwrap_self_sgmcmc, zip([self]*ncores, [int(niter/ncores)]*ncores,[int(burnin/ncores)]*ncores,[batch_size]*ncores, multi_backend,rng))

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
        '''
        pool = Pool(processes=ncores)
        results= pool.map(unwrap_self_mean, zip([self]*ncores, multi_backend))
        aux={var:((np.sum([r[var] for r in results],axis=0).reshape(self.start[var].shape))/niter) for var in self.start.keys()}
        return aux
        '''

    def sample_mean(self, filename):
        f=h5py.File(filename)
        aux = {var:np.sum(f[var],axis=0) for var in f.keys()}
        return aux
        