import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import sys
import time
import h5py
sys.path.append("./")

import hamiltonian.poisson as poisson
import hamiltonian.utils as utils

eta=1e-5
epochs=100
batch_size=100
alpha=1./100.

data_train=h5py.File('data/walmart/data_train_8_1.h5','r')
X_train=data_train['X'][:]
y_train=data_train['y'][:]

data_test=h5py.File('data/walmart/data_test_8_1.h5','r')
X_test=data_test['X'][:]
y_test=data_test['y'][:]

data_val=h5py.File('data/walmart/data_val_8_1.h5','r')
X_val=data_val['X'][:]
y_val=data_val['y'][:]

D=X_train.shape[1]

import time


start_p={'weights':np.zeros((D)),
        'bias':np.zeros(1)}
hyper_p={'alpha':alpha}

start_time=time.time()
par_sgd,loss_sgd=poisson.sgd(X_train,y_train,start_p,hyper_p,eta=eta,epochs=epochs,batch_size=batch_size,verbose=1)
elapsed_time=time.time()-start_time 
print('SGD, time:',elapsed_time)
y_pred=poisson.predict(X_test,par_sgd)
print(utils.mean_absolute_percentage_error(y_test,y_pred))
print "-----------------------------------------------------------"
y_pred_val=poisson.predict(X_val,par_sgd)
print(utils.mean_absolute_percentage_error(y_val,y_pred_val))
print "-----------------------------------------------------------"

data_train.close()
data_test.close()
data_val.close()
