from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import hamiltonian.utils as utils
import time

aaa = time.time()
K = 2
D=10
centers = [np.random.random_integers(0,10,D) for i in range(K)]
X, y = make_blobs(n_samples=1000, centers=centers, cluster_std=10,random_state=40)

y=utils.one_hot(y,K)

X = (X - X.mean(axis=0)) / X.std(axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#############################
 
import sys
sys.path.append("../") 
import numpy as np
import time

gpu = False

alpha=1./10.
start_p={'weights':np.zeros((D,K)),'bias':np.zeros((K))}
hyper_p={'alpha':alpha}

if gpu:
    import hamiltonian.logisticgpu as logistic
    LOG=logistic.LOGISTIC()

    par,loss=LOG.sgd(X_train, y_train, start_p, hyper_p, eta=1e-5,epochs=1e4,batch_size=50,verbose=True)
else:
    #import hamiltonian.softmaxcpu as softmax
    import hamiltonian.softmaxgpu as softmax
    SOFT=softmax.SOFTMAX()
    par,loss=SOFT.sgd(X_train.copy(), y_train.copy(),K, start_p, hyper_p, eta=1e-5,epochs=1e4,batch_size=50,verbose=True)

y_pred=SOFT.predict(X_test.copy(),par)
print(classification_report(y_test.copy().argmax(axis=1), y_pred))
print(confusion_matrix(y_test.copy().argmax(axis=1), y_pred))

'''
print ('-------------------------------------------')
from sklearn.linear_model import LogisticRegression
softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=1/alpha,fit_intercept=True)
softmax_reg.fit(X_train.copy(),np.argmax(y_train.copy(),axis=1))
y_pred2 = softmax_reg.predict(X_test.copy())
print(classification_report(y_test.copy().argmax(axis=1), y_pred2))
print(confusion_matrix(y_test.copy().argmax(axis=1), y_pred2))
print ('-------------------------------------------')
'''

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

print("antes")
mcmc=hmc.HMC(X_train,y_train,SOFT.loss, SOFT.grad, par, alpha, path_length=1,verbose=0)
print("despues")
posterior_sample,logp_samples=mcmc.multicore_sample(niter,burnin,backend=backend, ncores=ncores)


if backend:
    par_mean = mcmc.multicore_mean(posterior_sample, niter, ncores=ncores)

    y_pred_mc=LOG.predict(X_test,par_mean)

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))


else:
    if gpu:
        par_mean={var:cp.mean(posterior_sample[var],axis=0).reshape(start_p[var].shape) for var in posterior_sample.keys()}
        par_var={var:cp.var(posterior_sample[var],axis=0).reshape(start_p[var].shape) for var in posterior_sample.keys()}

        y_pred2=LOG.predict(X_test.copy(),par_mean)
        print(y_pred2)

        print(classification_report(y_test.copy(), y_pred2))
        print(confusion_matrix(y_test.copy(), y_pred2))
    else:
        post_par={var:np.mean(posterior_sample[var],axis=0).reshape(start_p[var].shape) for var in posterior_sample.keys()}
        post_par_var={var:np.std(posterior_sample[var],axis=0).reshape(start_p[var].shape) for var in posterior_sample.keys()}
        y_pred=SOFT.predict(X_test.copy(),post_par)

        print(classification_report(y_test.copy().argmax(axis=1), y_pred))
        print(confusion_matrix(y_test.copy().argmax(axis=1), y_pred))
        
        '''
        print ('-------------------------------------------')
        from sklearn.linear_model import LogisticRegression
        softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=1/alpha,fit_intercept=True)
        softmax_reg.fit(X_train.copy(),np.argmax(y_train.copy(),axis=1))
        y_pred2 = softmax_reg.predict(X_test.copy())
        print(classification_report(y_test.copy().argmax(axis=1), y_pred2))
        print(confusion_matrix(y_test.copy().argmax(axis=1), y_pred2))
        print ('-------------------------------------------')
        '''