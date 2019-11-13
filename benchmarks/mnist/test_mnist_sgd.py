import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import sys
import time
import h5py
sys.path.append("./")

import hamiltonian.utils as utils
import hamiltonian.models.gpu.softmax as base_model
import hamiltonian.inference.gpu.sgd as inference

eta=1e-5
epochs=100
batch_size=100
alpha=1./100.
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
K=len(classes)
y_train=utils.one_hot(y_train[:],K)
y_test=utils.one_hot(y_test[:],K)
import time


start_p={'weights':np.zeros((D,K)),
        'bias':np.zeros((K))}
hyper_p={'alpha':alpha}

start_time=time.time()
model=base_model.softmax(hyper_p)
optim=inference.sgd(model,start_p,step_size=eta)
par,loss=optim.fit(epochs=epochs,batch_size=batch_size,gamma=0.9,X_train=X_train,y_train=y_train,verbose=True)
print('SGD, time:',time.time()-start_time)
y_pred=model.predict(X_test,par_sgd,prob=False,batchsize=100)
cnf_matrix_sgd=confusion_matrix(y_test[:].argmax(axis=1), y_pred)
print(classification_report(y_test[:].argmax(axis=1), y_pred))

mnist_test.close()
mnist_train.close()

