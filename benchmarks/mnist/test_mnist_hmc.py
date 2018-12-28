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
import hamiltonian.utils as utils

path_length=10
epochs=20
batch_size=200
alpha=1e-2
data_path = 'data/'

mnist_train=h5py.File('data/mnist_train.h5','r')
X_train=mnist_train['X_train'][:].reshape((-1,28*28))
X_train=X_train/255.
y_train=mnist_train['y_train']

mnist_test=h5py.File('data/mnist_test.h5','r')
X_test=mnist_test['X_test'][:].reshape((-1,28*28))
X_test=X_test/255.
y_test=mnist_test['y_test']


classes=np.unique(y_train)
D=X_train.shape[1]
num_classes=len(classes)
y_train=utils.one_hot(y_train[:],num_classes)
y_test=utils.one_hot(y_test[:],num_classes)
start_p={'weights':1e-3*np.random.randn(D,num_classes-1),
        'bias':1e-3*np.random.randn(num_classes-1)}
hyper_p={'alpha':alpha}
mcmc=hmc.HMC(X_train,y_train,softmax.loss, softmax.grad, start_p,hyper_p, path_length=2,step_size=2e-5,verbose=1)
t0=time.clock()
posterior_sample=mcmc.sample(1e3,1e2)
t1=time.clock()
print("Ellapsed Time : ",t1-t0)

post_par={var:np.mean(posterior_sample[var],axis=0).reshape(start_p[var].shape) for var in posterior_sample.keys()}
y_pred=softmax.predict(X_test,post_par)
print(classification_report(y_test.argmax(axis=1), y_pred))
print(confusion_matrix(y_test.argmax(axis=1), y_pred))

#b_cols=columns=['b1', 'b2','b3']
#w_cols=[]
#for i in range(1,13):
#    w_cols.append('w'+str(i))

#b_sample = pd.DataFrame(posterior_sample['bias'], columns=b_cols)
#w_sample = pd.DataFrame(posterior_sample['weights'],columns=w_cols)

#print(b_sample.describe())
#print(w_sample.describe())
#sns.distplot(b_sample['b1'])
#sns.distplot(b_sample['b2'])
#sns.distplot(b_sample['b3'])
#sns.pairplot(b_sample)
#plt.show()
