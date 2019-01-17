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

sys.path.append("./") 
import hamiltonian.softmax as softmax
import hamiltonian.sgld as sampler
import hamiltonian.utils as utils

path_length=10
epochs=20
batch_size=1000
alpha=1e-2
data_path = 'data/'

plants_train=h5py.File('data/train_features_labels.h5','r')
X_train=plants_train['train_features']
y_train=plants_train['train_labels']
plants_test=h5py.File('data/validation_features_labels.h5','r')
X_test=plants_test['validation_features']
y_test=plants_test['validation_labels']

classes=np.unique(y_train)
D=X_train.shape[1]
K=y_train.shape[1]
import time

start_p={'weights':np.zeros((D,K)),
        'bias':np.zeros((K))}
hyper_p={'alpha':alpha}

mcmc=sampler.SGLD(X_train,y_train,softmax.loss, softmax.grad, start_p,hyper_p, path_length=1,verbose=1)
t0=time.clock()
posterior_sample,logp_samples=mcmc.sample(1e1,1e1,batch_size=batch_size)
t1=time.clock()
print("Ellapsed Time : ",t1-t0)

import pickle
outfile=open('results/sgmcmc_plants.pkl','wb')
pickle.dump(posterior_sample,outfile)

post_par={var:np.mean(posterior_sample[var],axis=0).reshape(start_p[var].shape) for var in posterior_sample.keys()}
y_pred=softmax.predict(X_test,post_par)
print(classification_report(y_test[:].argmax(axis=1), y_pred))
print(confusion_matrix(y_test[:].argmax(axis=1), y_pred))

plants_train.close()
plants_test.close()
