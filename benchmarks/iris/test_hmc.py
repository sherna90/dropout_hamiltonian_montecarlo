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

sys.path.append("./") 
use_gpu=False
if use_gpu:
    import hamiltonian.softmax_gpu as softmax
else:
    import hamiltonian.softmax as softmax

import hamiltonian.hmc as hmc
import hamiltonian.metropolis as mc
import hamiltonian.utils as utils
from scipy import stats


alpha=1/4.
path_length=10
iris = datasets.load_iris()
data = iris.data  
labels = iris.target
classes=np.unique(iris.target)
X, y = iris.data, iris.target
num_classes=len(classes)
y=utils.one_hot(y,num_classes)
X = (X - X.mean(axis=0)) / X.std(axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,shuffle=True)

D=X_train.shape[1]

start_p={'weights':1e-3*np.random.randn(D,num_classes),
        'bias':1e-3*np.random.randn(num_classes)}
hyper_p={'alpha':alpha}
mcmc=hmc.HMC(X_train,y_train,softmax.loss, softmax.grad, start_p,hyper_p, path_length=path_length,verbose=1)
t0=time.clock()
posterior_sample=mcmc.multicore_sample(1e4,1e3,None,4)
t1=time.clock()
print("Ellapsed Time : ",t1-t0)

post_par={var:np.mean(posterior_sample[var],axis=0).reshape(start_p[var].shape) for var in posterior_sample.keys()}
y_pred=softmax.predict(X_test,post_par)
print(classification_report(y_test.argmax(axis=1), y_pred))
print(confusion_matrix(y_test.argmax(axis=1), y_pred))

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
#sns.pairplot(b_sample)
plt.show()
