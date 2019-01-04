import warnings
warnings.filterwarnings("ignore")

import numpy as np
from numpy.linalg import *
from utils import *

def grad(X,y,par,hyper):
    cov=hyper['cov']
    grad={}
    grad['mu']=-np.dot(par['mu'],inv(cov))
    return grad	
    
def loss(X, y, par,hyper):
    dim=par['mu'].shape[0]
    cov=hyper['cov']
    log_loss=-0.5*np.dot(par['mu'].T,inv(cov)).dot(par['mu'])
    log_loss+=np.log(1./np.sqrt(linalg.det(cov)))
    log_loss+=dim*0.5*np.log(2*np.pi)
    return log_loss
