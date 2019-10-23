import warnings
warnings.filterwarnings("ignore")

import numpy as np
from hamiltonian.utils import *
from copy import deepcopy
from tqdm import tqdm
import cupy as cp
from cupy.linalg import norm
import time

class sgd:

    def __init__(self,model,start_p,step_size=0.1):
        self.start={var:cp.asarray(start_p[var]) for var in start_p.keys()}
        self.step_size = step_size
        self.model = model

    def iterate_minibatches(self, X, y, batchsize):
        assert X.shape[0] == y.shape[0]
        for start_idx in range(0, X.shape[0] - batchsize + 1, batchsize):
            excerpt = slice(start_idx, start_idx + batchsize)
            yield cp.asarray(X[excerpt]),cp.asarray(y[excerpt])

    def fit(self,epochs=1,batch_size=1,gamma=0.9,**args):
        X=args['X_train']
        y=args['y_train']
        if 'verbose' in args:
            verbose=args['verbose']
        else:
            verbose=None
        epochs=int(epochs)
        loss_val=cp.zeros((cp.int(epochs)))
        par_gpu=deepcopy(self.start)
        momentum={var:cp.zeros_like(self.start[var]) for var in self.start.keys()}
        #n_data=cp.float(y.shape[0])
        for i in tqdm(range(np.int(epochs))):
            for X_batch, y_batch in self.iterate_minibatches(X, y, batch_size):
                n_batch=cp.float(y_batch.shape[0])
                grad_p=self.model.grad(par_gpu,X_train=X_batch,y_train=y_batch)
                for var in par_gpu.keys():
                    momentum[var] = gamma * momentum[var] - self.step_size * grad_p[var]
                    par_gpu[var]+=momentum[var]
            loss_val[i]=-1.*self.model.log_likelihood(par_gpu,X_train=X_batch,y_train=y_batch)
            if verbose and (i%(epochs/10)==0):
                print('loss: {0:.4f}'.format(cp.asnumpy(loss_val[i])))
        return par_gpu,loss_val

    def fit_dropout(self,epochs=1,batch_size=1,p=0.5,gamma=0.9,**args):
        X=args['X_train']
        y=args['y_train']
        if 'verbose' in args:
            verbose=args['verbose']
        else:
            verbose=None
        loss_val=cp.zeros((cp.int(epochs)))
        par_gpu=deepcopy(self.start)
        momemtum={var:cp.zeros_like(par_gpu[var]) for var in par_gpu.keys()}
        for i in range(int(epochs)):
            for batch in self.iterate_minibatches(X, y, batch_size):
                X_batch, y_batch = batch
                Z=cp.random.binomial(1,p,size=X_batch.shape)
                X_batch_dropout=cp.multiply(X_batch,Z)
                grad_p=self.model.grad(par_gpu,X_train=X_batch_dropout,y_train=y_batch)
                for var in par_gpu.keys():
                    momemtum[var] = gamma * momemtum[var] + - self.step_size * grad_p[var]
                    par_gpu[var]+=momemtum[var]
            loss_val[i]=-1.*self.model.log_likelihood(par_gpu,X_train=X_batch,y_train=y_batch)
            if verbose and (i%(epochs/10)==0):
                print('loss: {0:.4f}'.format(cp.asnumpy(loss_val[i])))
        return par_gpu,loss_val
