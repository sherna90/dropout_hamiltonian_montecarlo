import cupy as cp
import scipy as sp
import os
from hamiltonian.utils import *
from cupy.linalg import inv,norm
from copy import deepcopy
from multiprocessing import Pool,cpu_count
import os 

from tqdm import tqdm, trange
import h5py 
import time

class sgmcmc:

    def __init__(self,model, start_p, path_length=1.0,step_size=0.1,verbose=True):
        self.start={var:cp.asarray(start_p[var]) for var in start_p.keys()}
        self.step_size = step_size
        self.path_length = path_length
        self.model = model
        self.verbose=verbose
        self.mu = cp.log(10 * self.step_size)  # proposals are biased upwards to stay away from 0.
        self.target_accept = 0.65
        self.gamma = 0.05
        self.t0 = 10.0
        self.t = 0
        self.kappa = 0.75
        self.error_sum = 0.0
        self.log_averaged_step = 0.0

    def step(self,state,momentum,rng,**args):
        pass

    def iterate_minibatches(self, X, y, batchsize):
        assert X.shape[0] == y.shape[0]
        for start_idx in range(0, X.shape[0] - batchsize + 1, batchsize):
            excerpt = slice(start_idx, start_idx + batchsize)
            yield cp.asarray(X[excerpt]),cp.asarray(y[excerpt])

    def sample(self,epochs=1,burnin=1,batch_size=1,rng=None,**args):
        if rng == None:
            rng = cp.random.RandomState()
        X=args['X_train']
        y=args['y_train']
        if 'verbose' in args:
            verbose=args['verbose']
        else:
            verbose=None
        epochs=int(epochs)
        n_data=X.shape[0]
        num_batches=cp.ceil(y[:].shape[0]/float(batch_size))
        q,p=self.start,{var:cp.zeros_like(self.start[var]) for var in self.start.keys()}
        print('start burnin')
        for i in tqdm(range(int(burnin))):
            j=0
            for X_batch, y_batch in self.iterate_minibatches(X, y, batch_size):
                n_batch=X_batch.shape[0]
                kwargs={'X_train':X_batch,'y_train':y_batch,'verbose':verbose,'n_data':n_data,'n_batch':n_batch}
                q,p=self.step(q,p,rng,**kwargs)
                if (j % 1000)==0:
                    ll=-1.0*cp.asnumpy(self.model.log_likelihood(q,**kwargs))
                    print('burnin {0}, loss: {1:.4f}, mini-batch update : {2}'.format(i,ll,j))
                j=j+1
        logp_samples=cp.zeros(epochs)
        posterior={var:[] for var in self.start.keys()}
        print('start sampling')
        initial_step_size=self.step_size
        for i in tqdm(range(epochs)):
            j=0
            for X_batch, y_batch in self.iterate_minibatches(X, y, batch_size):
                n_batch=X_batch.shape[0]
                kwargs={'X_train':X_batch,'y_train':y_batch,'verbose':verbose,'n_data':n_data,'n_batch':n_batch}
                q,p=self.step(q,p,rng,**kwargs)
                self.step_size=self.lr_schedule(initial_step_size,j,num_batches)
                if (j % 1000 )==0:
                    ll=-1.0*cp.asnumpy(self.model.log_likelihood(q,**kwargs))
                    print('epoch {0}, loss: {1:.4f}, mini-batch update : {2}'.format(i,ll,j))
                j=j+1
            #initial_step_size=self.step_size
            ll=cp.asnumpy(-1.0*self.model.log_likelihood(q,**kwargs))
            logp_samples[i]=ll
            for var in self.start.keys():
                posterior[var].append(cp.asnumpy(q[var]))
        for var in self.start.keys():
            posterior[var]=np.array(posterior[var])
        return posterior,logp_samples

    def lr_schedule(self,initial_step_size,step,num_batches):
        decay_factor=initial_step_size/num_batches
        return initial_step_size * (1.0/(1.0+step*decay_factor*num_batches))