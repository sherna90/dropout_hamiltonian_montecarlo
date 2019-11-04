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
            K+=dim*cp.log(cp.sqrt(self.hyper['alpha']))
            K-=0.5*self.hyper['alpha']*cp.sum(cp.square(par[var]))
        return K
    

    def grad(self, par,**args):
        for k,v in args.items():
            if k=='X_train':
                X=cp.asarray(v)
            elif k=='y_train':
                y=cp.asarray(v)
        yhat=self.net(par,**args)
        diff = y.reshape(-1,1)-yhat
        #diff=diff[:,:-1]
        grad_w = cp.dot(X.T, diff)/float(y.shape[0])
        grad_b = cp.sum(diff, axis=0)/float(y.shape[0])
        grad={}
        grad['weights']=grad_w-self.hyper['alpha']*par['weights']
        grad['weights']=-1.0*grad['weights']
        grad['bias']=grad_b-self.hyper['alpha']*par['bias']
        grad['bias']=-1.0*grad['bias']
        return grad	

    def net(self,par,**args):
        for k,v in args.items():
            if k=='X_train':
                X=cp.asarray(v)
        y_linear = cp.dot(X, par['weights']) + par['bias']
        yhat = self.sigmoid(y_linear)
        return yhat

    def sigmoid(self, y_linear):
        norms=(1.0 + cp.exp(-y_linear))
        return 1.0 / norms

    def negative_log_posterior(self, par,**args ):
        return -1.0*(self.log_likelihood(par,**args)+self.log_prior(par,**args))

    def log_likelihood(self, par,**args):
        for k,v in args.items():
            if k=='X_train':
                X=cp.asarray(v)
            elif k=='y_train':
                y=cp.asarray(v)
        y_pred=self.net(par,**args)
        ll= cp.mean(np.multiply(y,cp.log(y_pred))+np.multiply((1.0-y),cp.log(1.0-y_pred)))
        return ll

    def predict(self, par,X,prob=False,batchsize=32):
        par_gpu={var:cp.asarray(par[var]) for var in par.keys()}
        results=[]
        for start_idx in range(0, X.shape[0] - batchsize + 1, batchsize):
            excerpt = slice(start_idx, start_idx + batchsize)
            X_gpu=cp.asarray(X[excerpt])  
            yhat=self.net(par_gpu,X_train=X_gpu)
            if prob:
                out=yhat
            else:
                out= (yhat > 0.5).astype(int).flatten()
            results.append(cp.asnumpy(out))
        results=np.asarray(results)
        return results.flatten()	

