import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cupy as cp
from utils import *
from copy import deepcopy
from numpy.linalg import norm
from scipy.special import logsumexp

def cross_entropy(y_linear, y):
    #return y*np.log(sigmoid(y_linear))+(1-y)*np.log((1-sigmoid(y_linear)))
    return -np.log(1.0+np.exp(y_linear)) + y*y_linear

def log_prior(par,hyper):
    logpdf=lambda z,alpha,k : -0.5*np.sum(alpha*np.square(z))-0.5*k*np.log(1./alpha)-0.5*k*np.log(2*np.pi)
    par_dims={var:np.array(par[var]).size for var in par.keys()}
    log_prior=[logpdf(par[var],hyper['alpha'],par_dims[var]) for var in par.keys()]
    return np.sum(log_prior)

def sigmoid(y_linear):
    norms=(1.0+np.exp(-y_linear))
    return 1.0 / norms

def sigmoid_gpu(y_linear):
    norms=(1.0+cp.exp(-y_linear))
    return 1.0 / norms

def net(X,par):
    y_linear = np.dot(X, par['weights']) + par['bias']
    yhat = sigmoid(y_linear)
    return yhat

def net_gpu(X,par):
    y_linear = cp.dot(X, par['weights']) + par['bias']
    yhat = sigmoid_gpu(y_linear)
    return yhat

def grad(X,y,par,hyper):
    yhat=net(X,par)
    diff = y-yhat
    grad_w = np.dot(X.T, diff)
    grad_b = np.sum(diff, axis=0)
    grad={}
    grad['weights']=grad_w
    grad['weights']+=hyper['alpha']*par['weights']
    grad['bias']=grad_b
    grad['bias']+=hyper['alpha']*par['bias']
    return grad	

def grad_gpu(X,y,par,hyper):
    yhat=net_gpu(X,par)
    diff = y-yhat
    grad_w = cp.dot(X.T, diff)
    grad_b = cp.sum(diff, axis=0).reshape(1)

    '''
    #AQUI ESTAEL ERROR EN EL SHAPE#
    aux = cp.sum(diff, axis=0)
    print aux
    print aux.shape
    aux = aux.reshape(1)
    print aux
    print aux.shape
    #print grad_b.shape
    #print (hyper['alpha']*cp.asarray(par['bias'])).shape
    '''

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
    #return log_likelihood(X, y, par,hyper)+log_prior(par,hyper)
    return log_likelihood(X, y, par,hyper)

def iterate_minibatches(X, y, batchsize):
    assert X.shape[0] == y.shape[0]
    for start_idx in range(0, X.shape[0] - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield X[excerpt], y[excerpt]

def iterate_minibatches_gpu(X_gpu, y_gpu, batchsize):
    assert X_gpu.shape[0] == y_gpu.shape[0]
    for start_idx in range(0, X_gpu.shape[0] - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield X_gpu[excerpt], y_gpu[excerpt]

def sgd(X, y, par,hyper,eta=1e-2,epochs=1e2,batch_size=20,verbose=True):
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
        loss_val[i]=-loss(X_batch,y_batch,par,hyper)/float(batch_size)
        if verbose and (i%(epochs/10)==0):
            print('iteration {0:5d} , loss: {1:.4f}'.format(i,loss_val[i]) )
    return par,loss_val

def sgd_gpu(X, y, par,hyper,eta=1e-2,epochs=1e2,batch_size=20,verbose=True):
    X_gpu = cp.asarray(X)
    y_gpu = cp.asarray(y)
    par_gpu={'weights':cp.asarray(par['weights']),'bias':cp.asarray(par['bias'])}
    loss_val = cp.zeros(np.int(epochs))
    momemtum={var:cp.zeros_like(par[var]) for var in par.keys()}
    gamma=0.9
    for i in range(np.int(epochs)):
        for batch_gpu in iterate_minibatches_gpu(X_gpu, y_gpu, batch_size):
            X_batch_gpu, y_batch_gpu = batch_gpu
            grad_p_gpu=grad_gpu(X_batch_gpu,y_batch_gpu,par_gpu,hyper)
    #print len(grad_p_gpu)
    '''
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
        loss_val[i]=-loss(X_batch,y_batch,par,hyper)/float(batch_size)
        if verbose and (i%(epochs/10)==0):
            print('iteration {0:5d} , loss: {1:.4f}'.format(i,loss_val[i]) )
    return par,loss_val
    '''
    return 1, 2

def predict(X,par):
    yhat=net(X,par)
    pred=1*np.array( yhat > 0.5)
    return pred	

def predict_gpu(X,par):
    yhat=net_gpu(X,par)
    pred=1*cp.array( yhat > 0.5)
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
