import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import sys
import time
import h5py
sys.path.append("./")
use_gpu=False
if use_gpu:
    import hamiltonian.softmax_gpu as softmax
else:
    import hamiltonian.softmax as softmax

eta=1e-2
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

classes=np.unique(y_train)

dim_data=X_train.shape[1]
num_classes=len(classes)
import time

start_time=time.time()
start_p={'weights':np.random.randn(dim_data,num_classes),'bias':np.zeros((num_classes)),'alpha':alpha}
hyper_p={'alpha':alpha}
par,loss=softmax.sgd(X_train,y_train,num_classes,start_p,hyper_p,eta=eta,epochs=epochs,batch_size=batch_size,scale=True,transform=True,verbose=1)
elapsed_time=time.time()-start_time 
print(elapsed_time)
y_pred=softmax.predict(X_test,par)

print(classification_report(y_test[:], y_pred))
print(confusion_matrix(y_test[:], y_pred))
mnist_test.close()
mnist_train.close()

