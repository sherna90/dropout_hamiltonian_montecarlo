import warnings
warnings.filterwarnings("ignore")

import numpy as np
from utils import *
from copy import deepcopy
from numpy.linalg import norm
from scipy.special import logsumexp

def cross_entropy(y_est, y,log_enc=False):
    if log_enc:
        lse=logsumexp(y_est,axis=1)
        y_hat=y_est-np.repeat(lse[:,np.newaxis],y.shape[1]).reshape(y.shape)
    else:
        y_hat = np.log(y_est)
    return np.sum(y *  y_hat,axis=1)

def log_prior(par,hyper):
    dim=par['weights'].size+par['bias'].size
    log_prior=0.5*dim*np.log(hyper['alpha'])-dim*np.log(2*np.pi)
    log_prior+=-0.5*hyper['alpha']*np.sum(np.square(par['weights']))
    log_prior+=-0.5*hyper['alpha']*np.sum(np.square(par['bias']))
    return log_prior

def softmax(y_linear):
    exp = np.exp(y_linear-np.max(y_linear, axis=1).reshape((-1,1)))
    norms = np.sum(exp, axis=1).reshape((-1,1))
    return exp / norms

def net(X,par):
    y_linear = np.dot(X, par['weights']) + par['bias']
    yhat = softmax(y_linear)
    return yhat

def grad(X,y,par,hyper):
    n_data=float(y.shape[0])
    yhat=net(X,par)
    diff = yhat-y
    grad_w = np.dot(X.T, diff)
    grad_b = np.sum(diff, axis=0)
    grad={}
    grad['weights']=grad_w/n_data
    grad['weights']+=hyper['alpha']*par['weights']/n_data
    grad['bias']=grad_b
    grad['bias']+=hyper['alpha']*par['bias']/n_data
    return grad	
  
def loss(X, y, par,hyper):
    #y_hat=net(X,par)
    y_linear = np.dot(X, par['weights']) + par['bias']
    log_like=np.sum(cross_entropy(y_linear,y,log_enc=True))
    return -(log_like+log_prior(par,hyper))

def iterate_minibatches(X, y, batchsize):
    assert X.shape[0] == y.shape[0]
    for start_idx in range(0, X.shape[0] - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield X[excerpt], y[excerpt]

def sgd(X, y,num_classes, par,hyper,eta=1e-2,epochs=1e2,batch_size=20,verbose=True):
    loss_val=np.zeros((np.int(epochs)))
    momemtum={var:np.zeros((par[var].shape)) for var in par.keys()}
    gamma=0.9
    for i in range(np.int(epochs)):
        for batch in iterate_minibatches(X, y, batch_size):
            X_batch, y_batch = batch
            grad_p=grad(X_batch,y_batch,par,hyper)
            for var in par.keys():
                momemtum[var] = gamma * momemtum[var] + eta * grad_p[var]
                par[var]-=momemtum[var]
        loss_val[i]=loss(X_batch,y_batch,par,hyper)/float(batch_size)
        if verbose and (i%(epochs/10)==0):
            print('loss: {0:.8f}'.format(loss_val[i]) )
    return par,loss_val

def predict(X,par):
    yhat=net(X,par)
    pred=yhat.argmax(axis=1)
    return pred	

def check_gradient(X, y, par,hyper,dh=0.00001):
    grad_a=grad(X,y,par,hyper)
    x_plus=deepcopy(par)
    x_minus=deepcopy(par)
    diff=deepcopy(par)
    grad_f=deepcopy(par)
    for var in par.keys():
        x_plus[var]+=dh
        x_minus[var]-=dh
        f_plus=loss(X,y,x_plus,hyper)
        f_minus=loss(X,y,x_minus,hyper)
        x_plus[var]=par[var]
        x_minus[var]=par[var]
        grad_f[var]=(f_plus-f_minus)/(2*dh) 
        diff[var]=norm(grad_f[var]-np.sum(grad_a[var]))
    return grad_f    
    #return np.max([d for d in diff.values()])
