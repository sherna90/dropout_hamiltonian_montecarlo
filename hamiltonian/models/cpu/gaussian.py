import warnings
warnings.filterwarnings("ignore")

import numpy as np
from numpy.linalg import *
from hamiltonian.utils import *
import hamiltonian.models.model as base_model

class gaussian:

    def __init__(self,_hyper):
        self.hyper=_hyper

    def grad(self,par,**args):
        sigma=self.hyper['sigma']
        mu=self.hyper['mu']
        x=par['x']
        grad={}
        grad['x']=(x - mu) / (sigma*sigma)
        return grad	
        
    def logp(self,par,**args):
        sigma=self.hyper['sigma']
        mu=self.hyper['mu']
        x=par['x']
        log_loss=0.5 * (np.log(2 * np.pi * sigma * sigma) + ((x - mu) / sigma) ** 2)
        return log_loss