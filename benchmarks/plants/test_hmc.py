import warnings
warnings.filterwarnings("ignore")

from sklearn import datasets
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
import sys 
import pandas as pd
import time
import h5py

sys.path.append("./") 
use_gpu=False
if use_gpu:
    import hamiltonian.softmax_gpu as softmax
else:
    import hamiltonian.softmax as softmax

import hamiltonian.hmc as hmc

eta=1e-2
epochs=20
batch_size=256
alpha=1e-3
data_path = 'data/'

plants_train=h5py.File(data_path+'train_features_labels.h5','r')
X_train=plants_train['train_features']
y_train=plants_train['train_labels']
plants_test=h5py.File(data_path+'validation_features_labels.h5','r')
X_test=plants_test['validation_features']
y_test=plants_test['validation_labels']

classes=np.unique(y_train)

dim_data=X_train.shape[1]
num_classes=38
import time


D=X_train.shape[1]
num_classes=len(classes)
start_p={'weights':np.random.randn(D,num_classes),
        'bias':np.random.randn(num_classes)}
hyper_p={'alpha':alpha}
mcmc=hmc.HMC(X_train,y_train,softmax.loss, softmax.grad, start_p,hyper_p, n_steps=10,scale=False,transform=False,verbose=1)
t0=time.clock()
posterior_sample=mcmc.sample(1e4,1e3)
t1=time.clock()
print("Ellapsed Time : ",t1-t0)
post_par=start_p={'weights':np.mean(posterior_sample['weights'],axis=0).reshape(start_p['weights'].shape),
    'bias':np.mean(posterior_sample['bias'],axis=0).reshape(start_p['bias'].shape)}
y_pred=softmax.predict(X_test,post_par,False)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

hf_train = h5py.File(data_path+'hmc_samples.h5', 'w')
hf_train.create_dataset('weights', data=posterior_sample['weights'])
hf_train.create_dataset('bias', data=posterior_sample['bias'])
hf_train.close()
