import warnings
warnings.filterwarnings("ignore")

from sklearn import datasets
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys 

use_gpu=False
if use_gpu:
    import hamiltonian.softmax_gpu as softmax
else:
    import hamiltonian.softmax as softmax

import hamiltonian.hmc as hmc

epochs = 100
eta=1e-2
batch_size=10
alpha=1e-3
scaler = StandardScaler()

iris = datasets.load_iris()
data = iris.data  
labels = iris.target
classes=np.unique(iris.target)
X, y = iris.data, iris.target
X=scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


D=X_train.shape[1]
num_classes=len(classes)
start_p={'weights':np.random.randn(D,num_classes),'bias':np.random.randn(num_classes),'alpha':alpha}
mcmc=hmc.HMC(X_train,y_train,softmax.loss, softmax.grad, start_p, n_steps=100,scale=False,transform=True,verbose=1)
mcmc.sample(1e4)
par=mcmc.state
y_pred=softmax.predict(X_test,par,False)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print par