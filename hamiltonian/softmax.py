import warnings
warnings.filterwarnings("ignore")

import numpy as np
from utils import *

def cross_entropy(y_hat, y):
    return np.mean(np.sum(y * np.log(y_hat),axis=1))

def softmax(y_linear):
    exp = np.exp(y_linear-np.max(y_linear, axis=1).reshape((-1,1)))
    norms = np.sum(exp, axis=1).reshape((-1,1))
    return exp / norms

def net(X,par):
    ylinear = np.dot(X, par['weights']) + par['bias']
    yhat = softmax(ylinear)
    return yhat

def grad(X,y,par,hyper):
    n_data=float(y.shape[0])
    yhat=net(X,par)
    diff = yhat-y
    grad_w = np.dot(X.T, diff)
    grad_b = np.mean(diff, axis=0)
    grad={}
    grad['weights']=grad_w/n_data
    grad['weights']+=hyper['alpha']*par['weights']
    grad['bias']=grad_b
    grad['bias']+=hyper['alpha']*par['bias']
    return grad	
    
def loss(X, y, par,hyper):
    y_hat=net(X,par)
    n_data=float(y.shape[0])
    dim=par['weights'].shape
    log_like=cross_entropy(y_hat,y)
    log_prior=0.5*dim[0]*np.log(hyper['alpha'])
    log_prior+=0.5*dim[1]*np.log(hyper['alpha'])
    log_prior+=-0.5*hyper['alpha']*np.sum(np.square(par['weights']))
    log_prior+=-0.5*hyper['alpha']*np.sum(np.square(par['bias']))
    return log_like+log_prior

def iterate_minibatches(X, y, batchsize):
    assert X.shape[0] == y.shape[0]
    for start_idx in range(0, X.shape[0] - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield X[excerpt], y[excerpt]

def sgd(X, y,num_classes, par,hyper,eta=1e-2,epochs=1e2,batch_size=20,scale=True,transform=True,verbose=True):
    loss_val=np.zeros((np.int(epochs)))
    dim=par['weights'].shape[0]
    momemtum={'weights':np.zeros((par['weights'].shape)),'bias':np.zeros((par['bias'].shape))}
    gamma=0.9
    for i in range(np.int(epochs)):
        for batch in iterate_minibatches(X, y, batch_size):
            X_batch, y_batch = batch
            if scale:
                X_batch=X_batch/255.
                #X_batch,x_min,x_max=scaler_fit(X_batch)
            if transform:
                y_batch=one_hot(y_batch,num_classes)
            grad_p=grad(X_batch,y_batch,par,hyper)
            momemtum['weights'] = gamma * momemtum['weights'] + eta * grad_p['weights']
            par['weights']-=momemtum['weights']
            momemtum['bias'] = gamma * momemtum['bias'] + eta * grad_p['bias']    
            par['bias']-=momemtum['bias']
        loss_val[i]=-loss(X_batch,y_batch,par,hyper)
        if verbose and (i%(epochs/10)==0):
            print('loss: {0:.4f}'.format(loss_val[i]) )
    return par,loss_val

def predict(X,par,scale=False):
    if scale:
        X=X[:]/255.
        #X,x_min,x_max=scaler_fit(X[:])
    yhat=net(X,par)
    pred=yhat.argmax(axis=1)
    return pred	