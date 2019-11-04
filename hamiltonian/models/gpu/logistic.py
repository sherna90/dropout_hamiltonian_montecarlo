import warnings
warnings.filterwarnings("ignore")

import numpy as np
from hamiltonian.utils import *
from copy import deepcopy
from numpy.linalg import norm
from scipy.special import logsumexp
import cupy as cp
from tqdm import tqdm
import time

class logistic:
    
    def __init__(self,_hyper):
        self.hyper={var:cp.asarray(_hyper[var]) for var in _hyper.keys()}

    def log_prior(self, par,**args):
        for k,v in args.items():
            if k=='y_train':
                y=v
        K=0
        for var in par.keys():
            dim=(cp.asarray(par[var])).size
            K-=0.5*dim*cp.log(2*np.pi)
            K+=0.5*dim*cp.log(self.hyper['alpha'])
            K-=0.5*self.hyper['alpha']*cp.sum(cp.square(par[var]))
        return K
    

    def grad(self, par,**args):
        for k,v in args.items():
            if k=='X_train':
                X=cp.asarray(v)
            elif k=='y_train':
                y=cp.asarray(v)
        yhat=self.net(par,X)
        diff = y-yhat
        #diff=diff[:,:-1]
        grad_w = cp.dot(X.T, diff)
        grad_b = cp.sum(diff, axis=0)
        grad={}
        grad['weights']=grad_w+self.hyper['alpha']*par['weights']
        grad['weights']=-1.0*grad['weights']/float(y.shape[0])
        grad['bias']=grad_b+self.hyper['alpha']*par['bias']
        grad['bias']=-1.0*grad['bias']/float(y.shape[0])
        return grad	

    def net(self,par,**args):
        for k,v in args.items():
            if k=='X_train':
                X=cp.asarray(v)
            elif k=='y_train':
                y=cp.asarray(v)
        y_linear = cp.dot(X, par['weights']) + par['bias']
        yhat = self.sigmoid(y_linear)
        return yhat

    def sigmoid(self, y_linear):
        norms=(1.0 + cp.exp(-y_linear))
        return 1.0 / norms

    def loss(self, par,**args ):
        return self.log_likelihood(par,**args)

    def log_likelihood(self, par,**args):
        for k,v in args.items():
            if k=='X_train':
                X=cp.asarray(v)
            elif k=='y_train':
                y=cp.asarray(v)
        y_linear = cp.dot(X, par['weights']) + par['bias']
        ll= cp.sum()
        return ll

    def predict(self, X, par):
        X_gpu = cp.asarray(X)
        yhat = self.net(X_gpu, par)
        pred = 1 * cp.array( yhat > 0.5)
        return cp.asnumpy(pred)

