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
        h = F.relu(F.dropout(self.l1(x),ratio=.1))
        h = F.relu(F.dropout(self.l2(h),ratio=.1))
        return self.l3(F.dropout(h,ratio=.1))

class mlp():

    def __init__(self,_hyper,n_in, n_mid_units, n_out):
        self.hyper={var:cp.asarray(_hyper[var]) for var in _hyper.keys()}
        self.net=MyNetwork(n_in, n_mid_units, n_out)
        self.net.to_gpu()

    def log_prior(self, par,**args):
        K=0
        for var in par.keys():
            dim=(cp.asarray(par[var])).size
            K-=0.5*self.hyper['alpha']*cp.sum(cp.square(par[var]))/dim
        return K

    def grad(self, par,**args):
        for k,v in args.items():
            if k=='X_train':
                X=cp.asarray(v)
            elif k=='y_train':
                y=cp.asarray(v).astype(cp.int)
        self.net.enable_update()
        for param in self.net.namedparams():
            param[1].data=par[param[0]]
        y_hat = self.net(X)
        loss = F.softmax_cross_entropy(y_hat, y)
        # Calculate the gradients in the network
        self.net.cleargrads()
        loss.backward()
        grad={}
        for param in self.net.namedparams():
            grad[param[0]]=cp.asarray(param[1].grad)+0.5*self.hyper['alpha']*param[1].data
        return grad	
    
    def log_likelihood(self,par,**args):
        for k,v in args.items():
            if k=='X_train':
                X=cp.asarray(v)
            elif k=='y_train':
                y=cp.asarray(v).astype(cp.int)
        self.net.enable_update()
        for param in self.net.namedparams():
            param[1].data=par[param[0]]
            #param[1].data=cp.random.uniform(-1,1, param[1].shape,dtype=param[1].dtype)
        y_hat = self.net(X)
        loss = F.softmax_cross_entropy(y_hat, y)
        return loss.array
        
    def negative_log_posterior(self,par,**args):
        #return (self.log_likelihood(par,**args)+self.log_prior(par,**args))
        return self.log_likelihood(par,**args)+self.log_prior(par,**args)

    def predict(self,par,X_test,prob=False):
        X=cp.asarray(X_test)
        for param in self.net.namedparams():
            param[1].data=par[param[0]]
        y_hat = self.net(X)
        y_hat=y_hat.data
        y_hat=cp.asnumpy(y_hat)
        if prob:
            y = F.softmax(y_hat, axis=1)
            return y
        else:
            pred_label = y_hat.argmax(axis=1)
            return pred_label