import cupy as cp
import scipy as sp
import os
from hamiltonian.utils import *
from cupy.linalg import inv,norm
from copy import deepcopy
from multiprocessing import Pool,cpu_count
import os 

from hamiltonian.inference.gpu.sgmcmc import sgmcmc

from tqdm import tqdm, trange
import h5py 
import time

class sghmc(sgmcmc):
    

    def step(self,state,momentum,rng,**args):
        q = state.copy()
        p = self.draw_momentum(rng)
        q_new = deepcopy(q)
        p_new = deepcopy(p)
        epsilon=self.step_size
        path_length=cp.ceil(2*cp.random.rand()*self.path_length/epsilon)
        grad_q=self.model.grad(q,**args)
        # SG-HMC leapfrog step 
        for _ in cp.arange(path_length-1):
            for var in self.start.keys():
                dim=(cp.array(q_new[var])).size
                rvar=rng.normal(0,2*epsilon,dim).reshape(q[var].shape)
                q_new[var]+= epsilon*p_new[var]
                grad_q=self.model.grad(q_new,**args)
                p_new[var] = (1-epsilon)*p_new[var] + epsilon*grad_q[var]+rvar 
        acceptprob = self.accept(q, q_new, p, p_new,**args)
        if cp.isfinite(acceptprob) and (cp.random.rand() < acceptprob): 
            q = q_new.copy()
            p = p_new.copy()
        return q,p,acceptprob