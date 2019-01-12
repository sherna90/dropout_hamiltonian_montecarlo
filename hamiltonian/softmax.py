import warnings
warnings.filterwarnings("ignore")

import numpy as np
from utils import *
from copy import deepcopy
from numpy.linalg import norm
from scipy.special import logsumexp

def cross_entropy(y_linear, y):
    #y_linear=np.hstack((y_linear,np.zeros((y_linear.shape[0],1))))
    lse=logsumexp(y_linear,axis=1)
    y_hat=y_linear-np.repeat(lse[:,np.newaxis],y.shape[1]).reshape(y.shape)
    return np.sum(y *  y_hat,axis=1)

def log_prior(par,hyper):
    logpdf=lambda z,alpha,k : -0.5*np.sum(alpha*np.square(z))-0.5*k*np.log(1./alpha)-0.5*k*np.log(2*np.pi)
    par_dims={var:np.array(par[var]).size for var in par.keys()}
    log_prior=[logpdf(par[var],hyper['alpha'],par_dims[var]) for var in par.keys()]
    return np.sum(log_prior)

def softmax(y_linear):
    #y_linear=np.hstack((y_linear,np.zeros((y_linear.shape[0],1))))
    exp = np.exp(y_linear-np.max(y_linear, axis=1).reshape((-1,1)))
    norms = np.sum(exp, axis=1).reshape((-1,1))
    return exp / norms

def net(X,par):
    y_linear = np.dot(X, par['weights']) + par['bias']
    yhat = softmax(y_linear)
    return yhat

def grad(X,y,par,hyper):
    yhat=net(X,par)
    diff = y-yhat
    #diff=diff[:,:-1]
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
    ll= np.sum(cross_entropy(y_linear,y))
    return ll
    
def loss(X, y, par,hyper):
   return (log_likelihood(X, y, par,hyper)+log_prior(par,hyper))

def iterate_minibatches(X, y, batchsize):
    assert X.shape[0] == y.shape[0]
    for start_idx in range(0, X.shape[0] - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield X[excerpt], y[excerpt]

def sgd(X, y,num_classes, par,hyper,eta=1e-2,epochs=1e2,batch_size=150,verbose=True):
    loss_val=np.zeros((np.int(epochs)))
    momemtum={var:np.zeros_like(par[var]) for var in par.keys()}
    gamma=0.9
    for i in range(np.int(epochs)):
        for batch in iterate_minibatches(X, y, batch_size):
            X_batch, y_batch = batch
            grad_p=grad(X_batch,y_batch,par,hyper)
            for var in par.keys():
                momemtum[var] = gamma * momemtum[var] + eta * grad_p[var]/y_batch.shape[0]
                par[var]+=momemtum[var]
        loss_val[i]=-loss(X,y,par,hyper)/float(y.shape[0])
        if verbose and (i%(epochs/10)==0):
            print('loss: {0:.4f}'.format(loss_val[i]) )
    return par,loss_val

def predict(X,par,prob=False):
    yhat=net(X,par)
    pred=yhat.argmax(axis=1)
    if prob:
        out=yhat
    else:
        out=pred
    return out	

def sgd_dropout(X, y,num_classes, par,hyper,eta=1e-2,epochs=1e2,batch_size=150,verbose=True,p=0.5):
    loss_val=np.zeros((np.int(epochs)))
    momemtum={var:np.zeros_like(par[var]) for var in par.keys()}
    gamma=0.9
    for i in range(int(epochs)):
        for batch in iterate_minibatches(X, y, batch_size):
            X_batch, y_batch = batch
            n_x,n_y=X_batch.shape
            Z=np.random.binomial(1,p,n_x*n_y).reshape((n_x,n_y))
            X_batch_dropout=np.multiply(X_batch,Z)
            grad_p=grad(X_batch_dropout,y_batch,par,hyper)
            for var in par.keys():
                momemtum[var] = gamma * momemtum[var] + eta * grad_p[var]/y_batch.shape[0]
                par[var]+=momemtum[var]
        loss_val[i]=-loss(X,y,par,hyper)/float(y.shape[0])
        if verbose and (i%(epochs/10)==0):
            print('loss: {0:.4f}'.format(loss_val[i]) )
    return par,loss_val

def predict_stochastic(X,par,prob=False,p=0.5):
    n_x,n_y=X.shape
    Z=np.random.binomial(1,p,n_x*n_y).reshape((n_x,n_y))
    X_t=np.multiply(X,Z)   
    yhat=net(X_t,par)
    if prob:
        out=yhat
    else:
        out=yhat.argmax(axis=1)
    return out	


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
