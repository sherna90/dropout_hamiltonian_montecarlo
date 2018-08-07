
import numpy as np
import scipy as sp
import os


class Hamiltonian:
    def __init__(self, logp, gradf, start, step_size=1, n_steps=5):

        """ Hamiltonian MCMC sampler. Uses the gradient of log P(theta) to
            make informed proposals.
            Arguments
            ----------
            logp: function
                log P(X) function for sampling distribution
            start: dict
                Dictionary of starting state for the sampler. Should have one
                element for each argument of logp. So, if logp = f(x, y), then
                start = {'x': x_start, 'y': y_start}
            Keyword Arguments
            -----------------
            grad_logp: function or list of functions
                Functions that calculate grad log P(theta). Pass functions
                here if you don't want to use autograd for the gradients. If
                logp has multiple parameters, grad_logp must be a list of
                gradient functions w.r.t. each parameter in logp.
            scale: dict
                Same format as start. Scaling for initial momentum in
                Hamiltonian step.
            step_size: float
                Step size for the deterministic proposals.
            n_steps: int 
                Number of deterministic steps to take for each proposal.
            """
        self.start = start
        self.step_size = step_size / (len(self.start)**(1/4)
        self.n_steps = n_steps
        self.logp = logp
        self.state = {}


    def step(self):
        x = self.state
        r0 = initial_momentum()
        y, r = x, r0

        for i in range(self.n_steps):
            y, r = leapfrog(y, r, self.step_size, self.model.grad)

        if accept(x, y, r0, r, self.model.logp):
            x = y
            self._accepted += 1

        self.state = x
        self._sampled += 1
        return x

    @property
    def acceptance_rate(self):
        return self._accepted/self._sampled


    def leapfrog(self,x, r, step_size, grad):
        r1 = r + step_size/2*grad(x)
        x1 = x + step_size*r1
        r2 = r1 + step_size/2*grad(x1)
        return x1, r2


    def accept(self,x, y, r_0, r, logp):
        E_new = energy(logp, y, r)
        E = energy(logp, x, r_0)
        A = np.min(np.array([0, E_new - E]))
        return (np.log(np.random.rand()) < A)


    def energy(self,logp, x, r):
        return logp(x) - 0.5*np.dot(r1, r1)


    def initial_momentum(self):
        new = {}
        for var in self.start:
            new[var]=np.random.randn(len(self.start[var]))
        return new

    def sample(niter=1e3):
        for i in range(niter):
