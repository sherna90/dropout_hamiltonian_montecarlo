from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np

import sys
sys.path.append("../")

import hamiltonian.utils as utils
import hamiltonian.softmaxcpu as softmax
import hamiltonian.sgldPlantsQueue as sampler
import h5py
import time

################################## PLANTS HDF5 ##################################
alpha=1e-3
data_path = '../data/'

plants_train=h5py.File(data_path+'train_features_labels.h5','r')
X_train=plants_train['train_features']
y_train=plants_train['train_labels']
plants_test=h5py.File(data_path+'validation_features_labels.h5','r')
X_test=plants_test['validation_features']
y_test=plants_test['validation_labels']

D=X_train.shape[1]
num_classes=y_train.shape[1]
start_p={'weights':np.random.randn(D,num_classes),
        'bias':np.random.randn(num_classes)}
hyper_p={'alpha':alpha}

aux1 = X_train.shape[0]
aux2 = y_train.shape[1]

#plants_train.close()
################################## PLANTS HDF5 ##################################

SOFT=softmax.SOFTMAX()
par,loss=SOFT.sgd(X_train, y_train,num_classes, start_p, hyper_p, eta=1e-5,epochs=1e0,batch_size=50,verbose=True)

y_pred=SOFT.predict(X_test,par)
print(classification_report(y_test[:].argmax(axis=1), y_pred))
print(confusion_matrix(y_test[:].argmax(axis=1), y_pred))
print ('-------------------------------------------')

mcmc=sampler.SGLD(aux1, aux2, SOFT.loss, SOFT.grad, start_p.copy(),hyper_p.copy(), path_length=1,verbose=0)

#backend = 'test_sghmc_'
backend = None
niter = 10
burnin = 10

posterior_sample,logp_samples=mcmc.multicore_sample(niter,burnin,batch_size=50, backend=backend, ncores=4)

if backend:
    par_mean = mcmc.multicore_mean(posterior_sample, niter)

    y_pred_mc=SOFT.predict(X_test.copy(),par_mean)

    print(classification_report(y_test[:].argmax(axis=1), y_pred_mc))
    print(confusion_matrix(y_test[:].argmax(axis=1), y_pred_mc))
else:
    post_par={var:np.mean(posterior_sample[var],axis=0).reshape(start_p[var].shape) for var in posterior_sample.keys()}
    #post_par_var={var:np.var(posterior_sample[var],axis=0).reshape(start_p[var].shape) for var in posterior_sample.keys()}
    y_pred=SOFT.predict(X_test,post_par)
    print(classification_report(y_test[:].argmax(axis=1), y_pred))
    print(confusion_matrix(y_test[:].argmax(axis=1), y_pred))

print ('-------------------------------------------')
from sklearn.linear_model import LogisticRegression
softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=1/alpha,fit_intercept=True)
softmax_reg.fit(X_train,np.argmax(y_train,axis=1))
y_pred2 = softmax_reg.predict(X_test)
print(classification_report(y_test[:].argmax(axis=1), y_pred2))
print(confusion_matrix(y_test[:].argmax(axis=1), y_pred2))