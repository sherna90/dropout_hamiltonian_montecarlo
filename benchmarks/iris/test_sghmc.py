import warnings
warnings.filterwarnings("ignore")

from sklearn import datasets
import numpy as np
#import cupy as cp
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
import sys 
import pandas as pd
import time

sys.path.append("../../") 
import hamiltonian.cpu.softmax as softmax
import hamiltonian.cpu.sgld_multicore as sampler
import hamiltonian.utils as utils

niter = 2e3
burnin = 300
backend = 'iris'
#backend = None

alpha=1./4.
path_length=1
iris = datasets.load_iris()
classes=np.unique(iris.target)
X, y = iris.data, iris.target
num_classes=len(classes)
y=utils.one_hot(y,num_classes)
X = (X - X.mean(axis=0)) / X.std(axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,shuffle=True)

D=X_train.shape[1]
num_classes=len(classes)
start_p={'weights':np.random.randn(D,num_classes),
        'bias':np.random.randn(num_classes)}
hyper_p={'alpha':alpha}

model=softmax.SOFTMAX()
inference=sampler.sgld_multicore(model.log_likelihood, model.grad, start_p,hyper_p, path_length=path_length,verbose=0)
t0=time.time()
posterior_sample,logp_samples=inference.multicore_sample(X_train,y_train,niter=niter,burnin=burnin,backend=backend, ncores=2)
t1=time.time()
print("Ellapsed Time : ",t1-t0)

if backend:
    par_mean = inference.backend_mean(posterior_sample, niter)

    y_pred_mc=model.predict(X_test.copy(),par_mean)

    print(classification_report(y_test.copy().argmax(axis=1), y_pred_mc))
    print(confusion_matrix(y_test.copy().argmax(axis=1), y_pred_mc))
else:
    post_par={var:np.mean(posterior_sample[var],axis=0).reshape(start_p[var].shape) for var in posterior_sample.keys()}
    #post_par_var={var:np.var(posterior_sample[var],axis=0).reshape(start_p[var].shape) for var in posterior_sample.keys()}
    y_pred=model.predict(X_test.copy(),post_par)
    print(classification_report(y_test.copy().argmax(axis=1), y_pred))
    print(confusion_matrix(y_test.copy().argmax(axis=1), y_pred))

'''
b_cols=columns=['b1', 'b2','b3']
w_cols=[]
for i in range(1,13):
    w_cols.append('w'+str(i))

b_sample = pd.DataFrame(posterior_sample['bias'], columns=b_cols)
w_sample = pd.DataFrame(posterior_sample['weights'],columns=w_cols)

print(b_sample.describe())
print(w_sample.describe())
sns.distplot(b_sample['b1'])
sns.distplot(b_sample['b2'])
sns.distplot(b_sample['b3'])
sns.pairplot(b_sample)
plt.show()
'''