from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np

D=2
centers = [np.random.random_integers(0,10,D),np.random.random_integers(0,10,D)]
X, y = make_blobs(n_samples=100, centers=centers, cluster_std=10,random_state=40)
X = (X - X.mean(axis=0)) / X.std(axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#############################
 
import sys
sys.path.append("../") 
import numpy as np
import time

gpu = True
alpha=1./4.
start_p={'weights':np.zeros(D),'bias':np.zeros(1)}
hyper_p={'alpha':alpha}

if gpu:
    import hamiltonian.logisticgpu as logistic
    LOG=logistic.LOGISTIC()

    par,loss,numpy_par=LOG.sgd(X_train, y_train, start_p, hyper_p, eta=1e-5,epochs=1e4,batch_size=50,verbose=True)
else:
    import hamiltonian.logisticcpu as logistic
    LOG=logistic.LOGISTIC()

    par,loss=LOG.sgd(X_train, y_train, start_p, hyper_p, eta=1e-5,epochs=1e4,batch_size=50,verbose=True)

y_pred=LOG.predict(X_test.copy(), numpy_par)
print(y_pred)
print(classification_report(y_test.copy(), y_pred))
print(confusion_matrix(y_test.copy(), y_pred))

###############################

from multiprocessing import Pool,cpu_count

if gpu:
    import hamiltonian.hmcgpu as hmc
else:
    import hamiltonian.hmccpu as hmc

import hamiltonian.utils as utils
import numpy as np
import cupy as cp
import h5py 
import time

ncores=cpu_count()
#backend = 'simulated_data'
backend = None
niter = 1e3
burnin = 1e2

mcmc=hmc.HMC(X_train,y_train,LOG.loss, LOG.grad, numpy_par, alpha, path_length=1,verbose=0)

posterior_sample,logp_samples=mcmc.multicore_sample(niter,burnin,backend=backend, ncores=ncores)

#rng = [np.random.RandomState(i) for i in range(1)]
#posterior_sample,logp_samples=mcmc.sample(niter,burnin,backend, rng[0])

if backend:
    par_mean = mcmc.multicore_mean(posterior_sample, niter, ncores=ncores)

    y_pred_mc=LOG.predict(X_test,par_mean)

    print (y_pred_mc)
    print (y_pred)

else:
    if gpu:
        par_mean={var:cp.mean(posterior_sample[var],axis=0).reshape(start_p[var].shape) for var in posterior_sample.keys()}
        par_var={var:cp.var(posterior_sample[var],axis=0).reshape(start_p[var].shape) for var in posterior_sample.keys()}

        '''
        par_mean_gpu = start_p.copy()
        for var in par_mean.keys():
            for i in range(len(par_mean[var])):
                par_mean_gpu[var][i] = par_mean[var][i]

        par_mean_gpu={var:cp.asarray(par_mean_gpu[var]) for var in start_p.keys()}'''

        y_pred2=LOG.predict(X_test.copy(),par_mean)
        print(y_pred2)

        print(classification_report(y_test.copy(), y_pred2))
        print(confusion_matrix(y_test.copy(), y_pred2))
    else:
        par_mean={var:np.mean(posterior_sample[var],axis=0).reshape(start_p[var].shape) for var in posterior_sample.keys()}
        par_var={var:np.var(posterior_sample[var],axis=0).reshape(start_p[var].shape) for var in posterior_sample.keys()}
        y_pred=LOG.predict(X_test,par_mean)

        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))