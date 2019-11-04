import cupy as cp
from hamiltonian.utils import *
from cupy.linalg import inv,norm
from copy import deepcopy
from hamiltonian.inference.gpu.sgmcmc import sgmcmc


class sgld(sgmcmc):


    def step(self,state,momentum,rng,**args):
        for k,v in args.items():
            if k=='n_batch':
                n_batch=v
            elif k=='n_data':
                n_data=v
        epsilon=self.step_size
        q = deepcopy(state)
        nu = self.draw_momentum(rng,epsilon)
        p = deepcopy(momentum)
        grad_q=self.model.grad(q,**args)
        for var in p.keys():
            p[var]= nu[var]*p[var] - 0.5  * epsilon * grad_q[var]
            q[var]+=p[var]
        return q,p

    def draw_momentum(self,rng,epsilon):
        noise_scale = 2.0*epsilon
        sigma = np.sqrt(max(noise_scale, 1e-16))  
        momentum={var:rng.normal(0,noise_scale,size=self.start[var].shape) for var in self.start.keys()}
        return momentum


    