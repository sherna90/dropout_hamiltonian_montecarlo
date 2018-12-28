import warnings
warnings.filterwarnings("ignore")

import numpy as np
from utils import *
from copy import deepcopy
from numpy.linalg import norm
from scipy.special import logsumexp

def cross_entropy(y_linear, y):
    return np.log(1.0+np.exp(y_linear)) - y*y_linear

def log_prior(par,hyper):
    logpdf=lambda z,alpha,k : -0.5*np.sum(alpha*np.square(z))-0.5*k*np.log(1./alpha)-0.5*k*np.log(2*np.pi)
    par_dims={var:np.array(par[var]).size for var in par.keys()}
    log_prior=[logpdf(par[var],hyper['alpha'],par_dims[var]) for var in par.keys()]
    return np.sum(log_prior)

def sigmoid(y_linear):
    norms=(1.0+np.exp(-y_linear))
    return 1.0 / norms

def net(X,par):
    y_linear = np.dot(X, par['weights']) + par['bias']
    yhat = sigmoid(y_linear)
    return yhat

def grad(X,y,par,hyper):
    yhat=net(X,par)
    diff = yhat-y
    grad_w = np.dot(X.T, diff)
    grad_b = np.sum(diff, axis=0)
    grad={}
    grad['weights']=grad_w
    grad['weights']+=hyper['alpha']*par['weights']
    grad['bias']=grad_b
    grad['bias']+=hyper['alpha']*par['bias']
    return grad	
  
def log_likelihood(X, y, par,hyper):
    y_linear = np.dot(X, par['weights']) + par['bias']
    ll= -np.sum(cross_entropy(y_linear,y))
    return ll
    
def loss(X, y, par,hyper):
    return -(log_likelihood(X, y, par,hyper)+log_prior(par,hyper))

def iterate_minibatches(X, y, batchsize):
    assert X.shape[0] == y.shape[0]
    for start_idx in range(0, X.shape[0] - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield X[excerpt], y[excerpt]

def sgd(X, y, par,hyper,eta=1e-2,epochs=1e2,batch_size=20,verbose=True):
    loss_val=np.zeros((np.int(epochs)))
    momemtum={var:np.zeros((par[var].shape)) for var in par.keys()}
    gamma=1-0.99
    for i in range(np.int(epochs)):
        for batch in iterate_minibatches(X, y, batch_size):
            X_batch, y_batch = batch
            grad_p=grad(X_batch,y_batch,par,hyper)
            for var in par.keys():
                momemtum[var] = gamma * momemtum[var] + eta * grad_p[var]
                par[var]-=momemtum[var]
        loss_val[i]=loss(X_batch,y_batch,par,hyper)
        if verbose and (i%(epochs/10)==0):
            print('loss: {0:.4f}'.format(loss_val[i]) )
    return par,loss_val

def predict(X,par):
    yhat=net(X,par)
    pred=np.array( yhat > 0.5)
    return pred	


def check_gradient(X, y, par,hyper,dh=0.00001):
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
    #return np.max([d for d in diff.values()])
