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

import hamiltonian.sghmc as sghmc

path_length=10
epochs=20
batch_size=200
alpha=1e-2
data_path = 'data/'

mnist_train=h5py.File('data/mnist_train.h5','r')
X_train=mnist_train['X_train'][:].reshape((-1,28*28))
y_train=mnist_train['y_train']
mnist_test=h5py.File('data/mnist_test.h5','r')
X_test=mnist_test['X_test'][:].reshape((-1,28*28))
y_test=mnist_test['y_test']


D=X_train.shape[1]
num_classes=10
start_p={'weights':10*np.random.randn(D,num_classes),
        'bias':10*np.random.randn(num_classes)}
hyper_p={'alpha':alpha}
mcmc=sghmc.SGHMC(X_train,y_train,softmax.loss, softmax.grad, start_p,hyper_p, path_length=path_length,scale=True,transform=True,verbose=1)
t0=time.clock()
posterior_sample=mcmc.sample(10,1,10,20,backend='samples.h5')
t1=time.clock()
print("Ellapsed Time : ",t1-t0)


post_par={}

post_par['mean']={'weights':np.mean(posterior_sample['weights'],axis=0).reshape(start_p['weights'].shape),
    'bias':np.mean(posterior_sample['bias'],axis=0).reshape(start_p['bias'].shape)}
post_par['sd']={'weights':np.std(posterior_sample['weights'],axis=0).reshape(start_p['weights'].shape),
    'bias':np.std(posterior_sample['bias'],axis=0).reshape(start_p['bias'].shape)}


y_pred=softmax.predict(X_test,post_par['mean'],False)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

b_cols=[]
for i in range(1,num_classes+1):
    b_cols.append('b'+str(i))
w_cols=[]
for i in range(1,D*num_classes+1):
    w_cols.append('w'+str(i))

b_sample = pd.DataFrame(posterior_sample['bias'], columns=b_cols)
w_sample = pd.DataFrame(posterior_sample['weights'],columns=w_cols)

print(b_sample.describe())
print(w_sample.describe())

