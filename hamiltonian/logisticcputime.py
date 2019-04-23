import time as t
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from utils import *
from copy import deepcopy
from numpy.linalg import norm
from scipy.special import logsumexp

class LOGISTIC:
    def __init__(self, X_train, y_train , alpha, D):
        self.X=X_train
        self.y=y_train
        self.start_p = {'weights':np.zeros(D),'bias':np.zeros(1)}
        self.hyper_p = {'alpha':alpha}
        self.dot = []
        self.exp = []
        self.slices = []
        self.sum_grad = []
        self.dot_grad = []
        self.llenado = []
        self.logt = []
        self.time_e = 0

    def sgd(self, eta=1e-2,epochs=1e2,batch_size=20,verbose=True):
        var1 = t.time()
        loss_val = np.zeros(np.int(epochs))
        momemtum={var:np.zeros_like(self.start_p[var]) for var in self.start_p.keys()}
        gamma=0.9
        for i in range(np.int(epochs)):
            for batch in self.iterate_minibatches(batch_size):
                X_batch, y_batch = batch
                grad_p_gpu = self.grad(X_batch,y_batch)
                for var in self.start_p.keys():
                    momemtum[var] = gamma * momemtum[var] + eta * grad_p_gpu[var]/y_batch.shape[0]
                    self.start_p[var]+=momemtum[var]
            loss_val[i]=-self.loss(X_batch,y_batch)/float(batch_size)
            if verbose and (i%(epochs/10)==0):
                print('iteration {} , loss: {}'.format(i,loss_val[i]))
        self.time_e = t.time() - var1
        return self.start_p, loss_val

    def iterate_minibatches(self, batchsize):
        aux = t.time()
        assert self.X.shape[0] == self.y.shape[0]
        for start_idx in range(0, self.X.shape[0] - batchsize + 1, batchsize):
            excerpt = slice(start_idx, start_idx + batchsize)
            yield self.X[excerpt], self.y[excerpt]
        self.slices.append(t.time() - aux)


    def grad(self, X, y):
        aux = t.time()
        yhat=self.net(X)
        diff = y-yhat
        aux = t.time()
        grad_w = np.dot(X.T, diff)
        self.dot_grad.append(t.time() - aux)
        aux = t.time()
        grad_b = np.sum(diff, axis=0)
        self.sum_grad.append(t.time() - aux)
        grad={}
        aux = t.time()
        grad['weights']=grad_w
        grad['weights']+=self.hyper_p['alpha']*self.start_p['weights']
        grad['bias']=grad_b
        grad['bias']+=self.hyper_p['alpha']*self.start_p['bias']
        self.llenado.append(t.time() - aux)
        return grad	

    def net(self, X):
        aux = t.time()
        y_linear = np.dot(X, self.start_p['weights']) + self.start_p['bias']
        self.dot.append(t.time() - aux)
        yhat = self.sigmoid(y_linear)
        return yhat

    def sigmoid(self, y_linear):
        aux = t.time()
        norms=(1.0 + np.exp(-y_linear))
        self.exp.append(t.time() - aux)
        return 1.0 / norms

    def loss(self, X, y):
        return self.log_likelihood(X, y)

    def log_likelihood(self, X, y):
        y_linear = np.dot(X, self.start_p['weights']) + self.start_p['bias']
        ll= np.sum(self.cross_entropy(y_linear, y))
        return ll

    def cross_entropy(self, y_linear, y):
        aux1 = np.exp(y_linear)
        aux = t.time()
        var = -np.log(1.0 + aux1)
        self.logt.append(t.time() - aux)
        return -np.log(1.0 + np.exp(y_linear)) + y*y_linear

    def predict(self, X):
        yhat = self.net(X)
        pred = 1 * np.array( yhat > 0.5)
        return pred

    def stats(self):
        '''
        print "CPU DOT: ", np.mean(self.dot)
        print "CPU EXP: ", np.mean(self.exp)
        print "CPU SLICES: ", np.mean(self.slices)
        print "CPU DOT_GRAD: ", np.mean(self.dot_grad)
        print "CPU SUM_GRAD: ", np.mean(self.sum_grad)
        print "CPU LLENADO: ", np.mean(self.llenado)
        print "CPU LOGT: ", np.mean(self.logt)'''
        return np.mean(self.dot), np.mean(self.exp), np.mean(self.slices), np.mean(self.dot_grad), np.mean(self.sum_grad), np.mean(self.llenado), np.mean(self.logt), self.time_e