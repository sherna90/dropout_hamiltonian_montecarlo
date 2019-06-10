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

mcmc=sampler.SGLD(aux1, aux2, SOFT.loss, SOFT.grad, start_p.copy(),hyper_p.copy(), path_length=1,verbose=0)

niter = 10

posterior_sample = ["test_sghmc__%i.h5" %i for i in range(2)]

print(posterior_sample)

par_mean = mcmc.multicore_mean(posterior_sample, niter)
y_pred_mc=SOFT.predict(X_test,par_mean)

print(classification_report(y_test[:].argmax(axis=1), y_pred_mc))
print(confusion_matrix(y_test[:].argmax(axis=1), y_pred_mc))

print("Done!")
