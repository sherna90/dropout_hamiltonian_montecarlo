import warnings
warnings.filterwarnings("ignore")

import numpy as np
from hamiltonian.utils import *
from copy import deepcopy
from numpy.linalg import norm
import time

class logistic:
    
    def __init__(self,_hyper):
        self.hyper={var:np.asarray(_hyper[var]) for var in _hyper.keys()}

    def log_prior(self, par,**args):
        for k,v in args.items():
            if k=='y_train':
                y=v
        K=0
        for var in par.keys():
            dim=(np.asarray(par[var])).size
            K-=0.5*dim*np.log(2*np.pi)
            K+=dim*np.log(np.sqrt(self.hyper['alpha']))
            K-=0.5*self.hyper['alpha']*np.sum(np.square(par[var]))
        return K
    

    def grad(self, par,**args):
        for k,v in args.items():
            if k=='X_train':
                X=np.asarray(v)
            elif k=='y_train':
                y=np.asarray(v)
        yhat=self.net(par,**args)
        diff = y.reshape(-1,1)-yhat
        #diff=diff[:,:-1]
        grad_w = np.dot(X.T, diff)/float(y.shape[0])
        grad_b = np.sum(diff, axis=0)/float(y.shape[0])
        grad={}
        grad['weights']=grad_w-self.hyper['alpha']*par['weights']
        grad['weights']=-1.0*grad['weights']
        grad['bias']=grad_b-self.hyper['alpha']*par['bias']
        grad['bias']=-1.0*grad['bias']
        return grad	

    def net(self,par,**args):
        for k,v in args.items():
            if k=='X_train':
                X=np.asarray(v)
        y_linear = np.dot(X, par['weights']) + par['bias']
        yhat = self.sigmoid(y_linear)
        return yhat

    def sigmoid(self, y_linear):
        norms=(1.0 + np.exp(-y_linear))
        return 1.0 / norms

    def negative_log_posterior(self, par,**args ):
        return -1.0*(self.log_likelihood(par,**args)+self.log_prior(par,**args))

    def log_likelihood(self, par,**args):
        for k,v in args.items():
            if k=='X_train':
                X=np.asarray(v)
            elif k=='y_train':
                y=np.asarray(v)
        y_pred=self.net(par,**args)
        ll= np.mean(np.multiply(y,np.log(y_pred))+np.multiply((1.0-y),np.log(1.0-y_pred)))
        return ll


    def predict(self, par,X,prob=False,batchsize=32):
        par_gpu={var:np.asarray(par[var]) for var in par.keys()}
        results=[]
        for start_idx in range(0, X.shape[0] - batchsize + 1, batchsize):
            excerpt = slice(start_idx, start_idx + batchsize)
            X_batch=X[excerpt] 
            yhat=self.net(par_gpu,X_gpu)
            if prob:
                out=yhat
            else:
                out=(yhat>0.5).astype(int).flatten()
        results=np.asarray(results)
        return results.flatten()	


