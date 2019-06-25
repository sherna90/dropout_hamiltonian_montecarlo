import numpy as np
import scipy as sp
import os
from hamiltonian.utils import *
from numpy.linalg import inv
from copy import deepcopy
from multiprocessing import Pool,cpu_count
from tqdm import tqdm, trange
import h5py 
import os 
from scipy.optimize import check_grad
import math 
import time
import os
from hamiltonian.cpu.hmc import hmc

def unwrap_self_hmc(arg, **kwarg):
    return hmc_multicore.sample(*arg, **kwarg)

class hmc_multicore(hmc):
    
    def multicore_sample(self,X_train,y_train,niter=1e4,burnin=1e3,backend=None,ncores=cpu_count()):
        if backend:
            multi_backend = [backend+"_%i.h5" %i for i in range(ncores)]
        else:
            multi_backend = [backend]*ncores
    
        rng = [np.random.RandomState(i) for i in range(ncores)]

        pool = Pool(processes=ncores)
        results=pool.map(unwrap_self_hmc, zip([self]*ncores,[X_train]*ncores,[y_train]*ncores, [int(niter/ncores)]*ncores,[burnin]*ncores,multi_backend,rng))
        
        if not backend:
            posterior={var:np.concatenate([r[0][var] for r in results],axis=0) for var in self.start.keys()}
            logp_samples=np.concatenate([r[1] for r in results],axis=0)
            return posterior,logp_samples
        else:
            logp_samples=np.concatenate([r[1] for r in results],axis=0)
            return multi_backend,logp_samples