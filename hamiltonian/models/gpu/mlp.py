import math
import numpy as np
import chainer
from chainer import backend
from chainer import backends
from chainer.backends import cuda
from chainer import Function, FunctionNode, gradient_check, report, training, utils, Variable
from chainer import datasets, initializers, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.datasets import mnist
from chainer import iterators
from chainer import optimizers
import cupy as cp


class MyNetwork(Chain):

    def __init__(self,n_in, n_mid_units, n_out):
        super(MyNetwork, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(n_in, n_mid_units)
            self.l2 = L.Linear(n_mid_units, n_mid_units)
            self.l3 = L.Linear(n_mid_units, n_out)

    def forward(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        return self.l3(h)

class mlp():

    def __init__(self,_hyper,n_in, n_mid_units, n_out):
        self.hyper={var:cp.asarray(_hyper[var]) for var in _hyper.keys()}
        self.net=MyNetwork(n_in, n_mid_units, n_out)
        self.net.to_gpu(0)

    def log_prior(self, par,**args):
        K=0
        for var in par.keys():
            dim=(cp.asarray(par[var])).size
            #K-=0.5*self.hyper['alpha']*cp.sum(cp.square(par[var]))/dim
        return K

    def grad(self, par,**args):
        for k,v in args.items():
            if k=='X_train':
                X=cp.asnumpy(v)
            elif k=='y_train':
                y=cp.asnumpy(v)
                y=y.astype(np.int)
        for p in self.net.namedparams():
            p[1]=par[p[0]]
        y_hat = self.net(X)
        loss = F.softmax_cross_entropy(y_hat, y)
        # Calculate the gradients in the network
        self.net.cleargrads()
        loss.backward()
        grad={}
        for p in self.net.namedparams():
            grad[p[0]]=cp.asarray(p[1].grad)
        return grad	
    
    def log_likelihood(self,par,**args):
        for k,v in args.items():
            if k=='X_train':
                X=cp.asnumpy(v)
            elif k=='y_train':
                y=cp.asnumpy(v)
                y=y.astype(np.int)
        for p in self.net.namedparams():
            p[1]=par[p[0]]
        y_hat = self.net(X)
        loss = F.softmax_cross_entropy(X, y)
        return loss.array
        
    def negative_log_posterior(self,par,**args):
        return (self.log_likelihood(par,**args)+self.log_prior(par,**args))