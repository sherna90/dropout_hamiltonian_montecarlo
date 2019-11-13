import warnings
warnings.filterwarnings("ignore")

import numpy as np
from hamiltonian.utils import *
from copy import deepcopy
from tqdm import tqdm
import cupy as cp
from cupy.linalg import norm
import time

class softmax:

    def __init__(self,_hyper):
        self.hyper={var:cp.asarray(_hyper[var]) for var in _hyper.keys()}

    def logsumexp(self,log_prob,axis):
        max_prob = cp.max(log_prob,axis=axis)
        ds = log_prob - max_prob.reshape((log_prob.shape[0],1))
        sum_of_exp = cp.exp(ds).sum(axis=axis)
        return max_prob + cp.log(sum_of_exp)

    def cross_entropy(self, y_linear, y):
        #y_linear=cp.hstack((y_linear,cp.zeros((y_linear.shape[0],1))))
        lse=self.logsumexp(y_linear,axis=1)
        y_hat=y_linear-cp.repeat(lse[:,cp.newaxis],y.shape[1]).reshape(y.shape)
        return cp.sum(y *  y_hat,axis=1)

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
        return K/float(y.shape[0])

    def softmax(self, y_linear):
        exp = cp.exp(y_linear-cp.max(y_linear, axis=1).reshape((-1,1)))
        norms = cp.sum(exp, axis=1).reshape((-1,1))
        return exp / norms

    def net(self,par,X):
        y_linear = cp.dot(X, par['weights'])+par['bias']
        y_linear=cp.minimum(y_linear,-cp.log(cp.finfo(float).eps))
        y_linear=cp.maximum(y_linear,-cp.log(1./cp.finfo(float).tiny-1.0))
        yhat = self.softmax(y_linear)
        return yhat

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
        grad['weights']=-1.0*grad['weights']
        grad['bias']=grad_b+self.hyper['alpha']*par['bias']
        grad['bias']=-1.0*grad['bias']
        return grad	
    
    def log_likelihood(self,par,**args):
        for k,v in args.items():
            if k=='X_train':
                X=cp.asarray(v)
            elif k=='y_train':
                y=cp.asarray(v)
        y_linear = cp.dot(X, par['weights']) + par['bias']
        y_linear=cp.minimum(y_linear,-cp.log(cp.finfo(float).eps))
        y_linear=cp.maximum(y_linear,-cp.log(1./cp.finfo(float).tiny-1.0))
        return cp.sum(self.cross_entropy(y_linear,y))
        
    def negative_log_posterior(self,par,**args):
        for k,v in args.items():
            if k=='X_train':
                X=cp.asarray(v)
                n_data=X.shape[0]
        return (-1.0/n_data)*(self.log_likelihood(par,**args)+self.log_prior(par,**args))
    

    def predict(self, par,X,prob=False,batchsize=100):
        par_gpu={var:cp.asarray(par[var]) for var in par.keys()}
        results=[]
        for start_idx in range(0, X.shape[0] - batchsize + 1, batchsize):
            excerpt = slice(start_idx, start_idx + batchsize)
            X_gpu=cp.asarray(X[excerpt])  
            yhat=self.net(par_gpu,X_gpu)
            if prob:
                out=yhat
            else:
                out=yhat.argmax(axis=1)
            results.append(cp.asnumpy(out))
        results=np.asarray(results)
        dims=results.shape
        return results.reshape(dims[0]*dims[1],1)	


    def predict_stochastic(self,par,X,prob=False,p=0.5,batchsize=32):
        par_gpu={var:cp.asarray(par[var]) for var in par.keys()}
        results=[]
        for start_idx in range(0, X.shape[0] - batchsize + 1, batchsize):
            excerpt = slice(start_idx, start_idx + batchsize)
            X_gpu=cp.asarray(X[excerpt])  
            Z=cp.random.binomial(1,p,size=X_gpu.shape)
            X_t=cp.multiply(X_gpu,Z)   
            yhat=self.net(par_gpu,X_t)
            if prob:
                out=yhat
            else:
                out=yhat.argmax(axis=1)
            results.append(cp.asnumpy(out))
        return np.asarray(results)	


    