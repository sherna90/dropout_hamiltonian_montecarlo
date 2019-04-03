import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cupy as cp
from utils import *
from copy import deepcopy
from numpy.linalg import norm
from scipy.special import logsumexp

class LOGISTIC:
    def __init__(self, X_train, y_train , alpha, D):
        self.X=cp.asarray(X_train)
        self.y=cp.asarray(y_train)
        self.start_p = {'weights':cp.zeros(D),'bias':cp.zeros(1)}
        self.hyper_p = {'alpha':alpha}

    def sgd(self, eta=1e-2,epochs=1e2,batch_size=20,verbose=True):
        loss_val = cp.zeros(np.int(epochs))
        momemtum={var:cp.zeros_like(self.start_p[var]) for var in self.start_p.keys()}
        gamma=0.9
        for i in range(np.int(epochs)):
            for batch in self.iterate_minibatches(batch_size):
                X_batch, y_batch = batch
                grad_p_gpu = self.grad(X_batch,y_batch)
                #HASTA AQUI 3.5 SEGUNDOS#
                for var in self.start_p.keys():
                    momemtum[var] = gamma * momemtum[var] + eta * grad_p_gpu[var]/y_batch.shape[0]
                    self.start_p[var]+=momemtum[var]
                #HASTA AQUI 6.2 SEGUNDOS #
            aux = time.time()
            loss_val[i]=-self.loss(X_batch,y_batch)/float(batch_size)
            '''
            if verbose and (i%(epochs/10)==0):
                print('iteration {} , loss: {}'.format(i,loss_val[i]))
            '''
        return self.start_p, loss_val

    def iterate_minibatches(self, batchsize):
        assert self.X.shape[0] == self.y.shape[0]
        for start_idx in range(0, self.X.shape[0] - batchsize + 1, batchsize):
            excerpt = slice(start_idx, start_idx + batchsize)
            yield self.X[excerpt], self.y[excerpt]

    def grad(self, X, y):
        yhat=self.net(X)
        diff = y-yhat
        grad_w = cp.dot(X.T, diff)
        grad_b = cp.sum(diff, axis=0).reshape(1)
        grad={}
        grad['weights']=grad_w
        grad['weights']+=self.hyper_p['alpha']*self.start_p['weights']
        grad['bias']=grad_b
        grad['bias']+=self.hyper_p['alpha']*self.start_p['bias']
        return grad	

    def net(self, X):
        y_linear = cp.dot(X, self.start_p['weights']) + self.start_p['bias']
        yhat = self.sigmoid(y_linear)
        return yhat

    def sigmoid(self, y_linear):
        norms=(1.0 + cp.exp(-y_linear))
        return 1.0 / norms

    def loss(self, X, y):
        #return log_likelihood(X, y, par,hyper)+log_prior(par,hyper)
        return self.log_likelihood(X, y)

    def log_likelihood(self, X, y):
        y_linear = cp.dot(X, self.start_p['weights']) + self.start_p['bias']
        ll= cp.sum(self.cross_entropy(y_linear, y))
        return ll

    def cross_entropy(self, y_linear, y):
        #return y*np.log(sigmoid(y_linear))+(1-y)*np.log((1-sigmoid(y_linear)))
        return -cp.log(1.0 + cp.exp(y_linear)) + y*y_linear

    def predict(self, X):
        yhat = self.net(cp.asarray(X))
        pred = 1 * cp.array( yhat > 0.5)
        return cp.asnumpy(pred)