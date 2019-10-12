import numpy as np
import scipy as sp
import os
from numpy.linalg import inv,norm
from copy import deepcopy
import os 
from tqdm import tqdm, trange
import h5py 
import time

class sgd:
    
    def __init__(self,_model,eta=0.1,epochs=1,gamma=0.9,batch_size=1):
        self.eta=eta
        self.epochs=np.int(epochs)
        self.gamma=gamma
        self.batch_size=batch_size
        self.model=_model

    def iterate_minibatches(self, X, y, batchsize):
        assert X.shape[0] == y.shape[0]
        for start_idx in range(0, X.shape[0] - batchsize + 1, batchsize):
            excerpt = slice(start_idx, start_idx + batchsize)
            yield X[excerpt], y[excerpt]

    def fit(self,par,X, y,verbose=True):
        batch_size=self.batch_size
        loss_val=np.zeros(self.epochs)
        momentum={var:np.zeros_like(par[var]) for var in par.keys()}
        for i in tqdm(range(self.epochs)):
            for X_batch, y_batch in self.iterate_minibatches(X, y, self.batch_size):
                grad_p=self.model.grad(par,X_train=X_batch,y_train=y_batch)
                for var in par.keys():
                    momentum[var] = self.gamma * momentum[var] + self.eta * grad_p[var]
                    par[var]+=momentum[var]
            loss_val[i]=-1.*self.model.log_likelihood(par,X_train=X_batch,y_train=y_batch)
            if verbose and (i%(self.epochs/10)==0):
                print('loss: {0:.4f}'.format(loss_val[i]))
        return par,loss_val

    def fit_dropout(self,par,X, y,verbose=True,p=0.5):
        loss_val=np.zeros(self.epochs)
        momentum={var:np.zeros_like(par[var]) for var in par.keys()}
        for i in tqdm(range(self.epochs)):
            for X_batch, y_batch in self.iterate_minibatches(X, y, self.batch_size):
                n_batch=np.float(y_batch.shape[0])
                n_x,n_y=X_batch.shape
                Z=np.random.binomial(1,p,n_x*n_y).reshape((n_x,n_y))
                X_batch_dropout=np.multiply(X_batch,Z)
                grad_p=self.model.grad(par,X_train=X_batch_dropout,y_train=y_batch)
                for var in par.keys():
                    momentum[var] = self.gamma * momentum[var] + (1.0/n_batch)*self.eta * grad_p[var]
                    par[var]+=momentum[var]
            loss_val[i]=-1.*self.model.log_likelihood(par,X_train=X_batch,y_train=y_batch)
            if verbose and (i%(self.epochs/10)==0):
                print('loss: {0:.4f}'.format(loss_val[i]))
        return par,loss_val