import warnings
warnings.filterwarnings("ignore")

import cupy as cp
from cupy.linalg import *
from hamiltonian.utils import *
import hamiltonian.models.model as base_model

class mvn_gaussian:

    def __init__(self,_hyper):
        self.hyper={var:cp.asarray(_hyper[var]) for var in _hyper.keys()}

    def grad(self,par,**args):
        cov=self.hyper['cov']
        mu=self.hyper['mu']
        x=par['x']
        grad={}
        grad['x']=cp.dot(x-mu,inv(cov))
        return grad	
        
    def negative_log_posterior(self,par,**args):
        dim=self.hyper['mu'].shape[0]
        sigma=self.hyper['cov']
        mu=self.hyper['mu']
        x=par['x']
        log_loss=dim * cp.log(2 * cp.pi)
        log_loss+=cp.log(cp.linalg.det(sigma))
        log_loss+=cp.dot(np.dot((x - mu).T, cp.linalg.inv(sigma)), x - mu)
        log_loss*= 0.5
        return log_loss
