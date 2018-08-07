import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import sys
import time
import h5py

use_gpu=False
if use_gpu:
    import hamiltonian.softmax_gpu as softmax
else:
    import hamiltonian.softmax as softmax

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

start_time=time.time()
start_p={'weights':np.random.randn(dim_data,num_classes),'bias':np.random.randn(num_classes),'alpha':alpha}
par,loss=softmax.sgd(X_train,y_train,num_classes,start_p,eta=eta,epochs=epochs,batch_size=batch_size,scale=False,verbose=1)
elapsed_time=time.time()-start_time 
print(elapsed_time)
y_pred=softmax.predict(X_test,par,scale=False)
y_test_c=np.argmax(y_test[:],axis=1)
print(classification_report(y_test_c, y_pred))
print(confusion_matrix(y_test_c, y_pred))
plants_train.close()
plants_test.close()