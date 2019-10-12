import warnings
warnings.filterwarnings("ignore")

import numpy as np
from utils import *
from copy import deepcopy
from numpy.linalg import norm
from scipy.special import logsumexp
import time

class logistic:
    def __init__(self):
        pass

    def sgd(self,X, y, par,hyper, eta=1e-2,epochs=1e2,batch_size=20,verbose=True):
        
        loss_val = np.zeros(np.int(epochs))
        momemtum={var:np.zeros_like(par[var]) for var in par.keys()}
        gamma=0.9
        for i in range(np.int(epochs)):
            for batch in self.iterate_minibatches(X, y, batch_size):
                X_batch, y_batch = batch
                grad_p_gpu = self.grad(X_batch,y_batch,par,hyper)
                for var in par.keys():
                    momemtum[var] = gamma * momemtum[var] + eta * grad_p_gpu[var]/y_batch.shape[0]
                    par[var]+=momemtum[var]
            loss_val[i]=-self.loss(X_batch,y_batch,par,hyper)/float(batch_size)
            if verbose and (i%(epochs/10)==0):
                print('iteration {} , loss: {}'.format(i,loss_val[i]))
        return par, loss_val

    def iterate_minibatches(self,X, y, batchsize):
        assert X.shape[0] == y.shape[0]
        for start_idx in range(0, X.shape[0] - batchsize + 1, batchsize):
            excerpt = slice(start_idx, start_idx + batchsize)
            yield X[excerpt], y[excerpt]


    def grad(self, X,y,par,hyper):
        yhat=self.net(X, par)
        diff = y-yhat
        grad_w = np.dot(X.T, diff)
        grad_b = np.sum(diff, axis=0)
        grad={}
        grad['weights']=grad_w
        grad['weights']+=hyper['alpha']*par['weights']
        grad['bias']=grad_b
        grad['bias']+=hyper['alpha']*par['bias']
        return grad	

    def net(self, X,par):
        y_linear = np.dot(X, par['weights']) + par['bias']
        yhat = self.sigmoid(y_linear)
        return yhat

    def sigmoid(self, y_linear):
        norms=(1.0 + np.exp(-y_linear))
        return 1.0 / norms

    def loss(self, X, y, par,hyper):
        return self.log_likelihood(X, y, par,hyper)

    def log_likelihood(self, X, y, par,hyper):
        y_linear = np.dot(X, par['weights']) + par['bias']
        ll= np.sum(self.cross_entropy(y_linear, y))
        return ll

    def cross_entropy(self, y_linear, y):
        aux1 = np.exp(y_linear)
        var = -np.log(1.0 + aux1)
        return -np.log(1.0 + np.exp(y_linear)) + y*y_linear

    def predict(self, X, par):
        yhat = self.net(X, par)
        pred = 1 * np.array( yhat > 0.5)
        return pred

    def log_prior(self, par,hyper):
        logpdf=lambda z,alpha,k : -0.5*np.sum(alpha*np.square(z))-0.5*k*np.log(1./alpha)-0.5*k*np.log(2*np.pi)
        par_dims={var:np.array(par[var]).size for var in par.keys()}
        log_prior=[logpdf(par[var],hyper['alpha'],par_dims[var]) for var in par.keys()]
        return np.sum(log_prior)

    def check_gradient(self, X, y, par,hyper,dh=0.00001):
        grad_a=grad(X,y,par,hyper)
        x_plus=deepcopy(par)
        x_minus=deepcopy(par)
        diff={}
        grad_f={}
        for var in par.keys():
            x_plus[var]+=dh
            x_minus[var]-=dh
            f_plus=loss(X,y,x_plus,hyper)
            f_minus=loss(X,y,x_minus,hyper)
            x_plus[var]=par[var]
            x_minus[var]=par[var]
            grad_f[var]=(f_plus-f_minus)/(2.0*dh) 
            diff[var]=norm(grad_f[var]-np.sum(grad_a[var]))
        return grad_f   
