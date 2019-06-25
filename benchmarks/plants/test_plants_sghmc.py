import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn import datasets
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys 
import pandas as pd
import time
import h5py 

sys.path.append("../../") 
import hamiltonian.cpu.softmax as softmax
import hamiltonian.cpu.sghmc as sampler
import hamiltonian.utils as utils

path_length=10
epochs=20
batch_size=50
alpha=1e-2

data_path = '../../data/'

plants_train=h5py.File(data_path+'train_features_labels.h5','r')
X_train=plants_train['train_features']
y_train=plants_train['train_labels']
plants_test=h5py.File(data_path+'validation_features_labels.h5','r')
X_test=plants_test['validation_features']
y_test=plants_test['validation_labels']

classes=np.unique(y_train)
D=X_train.shape[1]
K=y_train.shape[1]
import time

start_p={'weights':np.zeros((D,K)),
        'bias':np.zeros((K))}
hyper_p={'alpha':alpha}

model=softmax.SOFTMAX()
mcmc=sampler.sghmc(model.loss, model.grad, start_p,hyper_p, path_length=1,verbose=1)
t0=time.clock()

#backend = "results/sgmcmc_plants"
backend = None
posterior_sample,logp_samples=mcmc.sample(X_train,y_train,1e3,1e2,batch_size,backend=backend)
t1=time.clock()
print("Ellapsed Time : ",t1-t0)

post_par={var:np.mean(posterior_sample[var],axis=0) for var in posterior_sample.keys()}
y_pred=softmax.predict(X_test,post_par)
print(classification_report(y_test[:].argmax(axis=1), y_pred))
print(confusion_matrix(y_test[:].argmax(axis=1), y_pred))

plants_train.close()
plants_test.close()
