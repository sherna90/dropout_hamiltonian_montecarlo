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

    def sgld_step(self,state,momentum,rng,**args):
        epsilon=self.step_size
        q = deepcopy(state)
        p = self.draw_momentum(rng,epsilon)
        grad_p=self.model.grad(q,**args)
        for var in p.keys():
            p[var]+=  - 0.5 * epsilon * grad_p[var]
            q[var]+=p[var]
        acceptprob=1.0
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
        num_batches=np.ceil(y[:].shape[0]/float(batch_size))
        decay_factor=self.step_size/num_batches
        #q,p=self.start,self.draw_momentum(rng)
        q,p=self.start,{var:np.zeros_like(self.start[var]) for var in self.start.keys()}
        print('start burnin')
        for i in tqdm(range(int(burnin))):
            for X_batch, y_batch in self.iterate_minibatches(X, y, batch_size):
                kwargs={'X_train':X_batch,'y_train':y_batch,'verbose':verbose}
                q,p,p_accept=self.sgld_step(q,p,rng,**kwargs)
                ll=-1.0*self.model.log_likelihood(q,**args)
                print('loss: {0:.4f}'.format(ll))
        logp_samples=np.zeros(epochs)
        posterior={var:[] for var in self.start.keys()}
        print('start sampling')
        initial_step_size=self.step_size
        for i in tqdm(range(epochs)):
            j=0
            for X_batch, y_batch in self.iterate_minibatches(X, y, batch_size):
                kwargs={'X_train':X_batch,'y_train':y_batch,'verbose':verbose}
                q,p,p_accept=self.sgld_step(q,p,rng,**kwargs)
                self.step_size=self.lr_schedule(initial_step_size,j,decay_factor,num_batches)
                ll=-1.0*self.model.log_likelihood(q,**args)
                print('loss: {0:.4f}, batch_update {1}'.format(ll,j))
                j=j+1
            #initial_step_size=self.step_size
            logp_samples[i]=self.model.logp(q,**args)
            for var in self.start.keys():
                posterior[var].append(q[var])
            if self.verbose and (i%(epochs/10)==0):
                print('loss: {0:.4f}'.format(logp_samples[i]))
        for var in self.start.keys():
            posterior[var]=np.array(posterior[var])
        return posterior,logp_samples

    def draw_momentum(self,rng,epsilon):
        momentum={}
        for var in self.start.keys():
            dim=(np.array(self.start[var])).size
            #rvar=rng.normal(0,self._inv_mass_matrix[var],dim)
            rvar=rng.normal(0,epsilon,dim)
            if dim>1:
                momentum[var]=rvar.reshape(self.start[var].shape)
            else:
                momentum[var]=rvar
        return momentum

    def lr_schedule(self,initial_step_size,step,decay_factor,num_batches):
        return initial_step_size * (1.0/(1.0+step*decay_factor*num_batches))