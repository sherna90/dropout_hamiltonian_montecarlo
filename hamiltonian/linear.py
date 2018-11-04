import warnings
warnings.filterwarnings("ignore")

import numpy as np
from utils import *


def net(X,par):
    yhat = np.dot(X, par['weights']) + par['bias']
    return yhat

def grad(X,y,par,hyper):
    n_data=float(y.shape[0])
    yhat=net(X,par)
    diff = yhat-y
    grad_w = np.dot(X.T, diff)
    grad_b = np.sum(diff, axis=0)
    grad={}
    grad['weights']=2.0*grad_w/n_data
    grad['weights']+=hyper['alpha']*par['weights']
    grad['bias']=2.0*grad_b/n_data
    grad['bias']+=hyper['alpha']*par['bias']
    return grad	
    
def loss(X, y, par,hyper):
    y_hat=net(X,par)
    n_data=float(y.shape[0])
    dim=par['weights'].shape
    log_like=-0.5*np.dot(par['mu'],inv(cov)).dot(par['mu'].T)
    log_like+=-np.log(1./np.sqrt(linalg.det(cov)))
    log_loss+=-dim*0.5*np.log(2*np.pi)
    log_like+=-0.5*hyper['alpha']*np.sum(np.square(par['weights']))
    log_like+=-0.5*hyper['alpha']*np.sum(np.square(par['bias']))
    return log_like/n_data

def predict(X,par,scale=False):
    if scale:
        X=X[:]/255.
        #X,x_min,x_max=scaler_fit(X[:])
    yhat=net(X,par)
    return yhat	