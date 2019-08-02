import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import sys
sys.path.append("../../") 
import time
import h5py
import pandas as pd
import hamiltonian.utils as utils
use_gpu=True
if use_gpu:
    import hamiltonian.gpu.softmax as softmax
else:
    import hamiltonian.cpu.softmax as softmax


import matplotlib.pyplot as plt 
import seaborn as sns
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Spectral):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(classes)
    plt.xticks(tick_marks, tick_marks, rotation=45,fontsize=8)
    plt.yticks(tick_marks, tick_marks,fontsize=8)
    fmt = '.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

eta=1e-3
epochs=100
batch_size=250
alpha=1e-2
data_path = '/home/sergio/data/PlantVillage-Dataset/balanced_train_test/features/'

plants_train=h5py.File(data_path+'plant_village_train.hdf5','r')
X_train=plants_train['features']
y_train=plants_train['labels']
plants_test=h5py.File(data_path+'plant_village_val.hdf5','r')
X_test=plants_test['features']
y_test=plants_test['labels']

D=X_train.shape[1]
K=y_train.shape[1]
import time

start_p={'weights':np.zeros((D,K)),
        'bias':np.zeros((K))}
hyper_p={'alpha':alpha}

start_time=time.time()
model=softmax.SOFTMAX()
par_sgd,loss_sgd=model.sgd(X_train,y_train,K,start_p,hyper_p,eta=eta,epochs=epochs,batch_size=batch_size,verbose=1)
elapsed_time=time.time()-start_time 
print('SGD, time:',elapsed_time)
y_pred=model.predict(X_test,par_sgd)
cnf_matrix_sgd=confusion_matrix(y_test[:].argmax(axis=1), y_pred)
print(classification_report(y_test[:].argmax(axis=1), y_pred))
print("-----------------------------------------------------------")
plants_train.close()
plants_test.close()
loss=pd.DataFrame(loss_sgd)
loss.to_csv('loss_sgd_gpu.csv',sep=',',header=False)
plt.figure()
plot_confusion_matrix(cnf_matrix_sgd, classes=np.int32(K),title='SGD GPU')
plt.savefig('plants_confusion_matrix_sgd_gpu.pdf',bbox_inches='tight')
plt.close()

""" start_time=time.time()
par_sgd_dropout_05,loss_sgd_dropout_05=softmax.sgd_dropout(X_train,y_train,K,start_p,hyper_p,eta=eta,epochs=epochs,batch_size=batch_size,verbose=0)
elapsed_time=time.time()-start_time 
print('SGD Dropout 0.5, time:',elapsed_time)
y_pred=softmax.predict(X_test,par_sgd_dropout_05)
cnf_matrix_dropout_05=confusion_matrix(y_test[:].argmax(axis=1), y_pred)
print(classification_report(y_test[:].argmax(axis=1), y_pred))
print "-----------------------------------------------------------"
start_time=time.time()
par_sgd_dropout_01,loss_sgd_dropout_01=softmax.sgd_dropout(X_train,y_train,K,start_p,hyper_p,eta=eta,epochs=epochs,batch_size=batch_size,verbose=0,p=0.1)
elapsed_time=time.time()-start_time 
print('SGD Dropout 0.1, time:',elapsed_time)
y_pred=softmax.predict(X_test,par_sgd_dropout_01)
cnf_matrix_dropout_01=confusion_matrix(y_test[:].argmax(axis=1), y_pred)
print(classification_report(y_test[:].argmax(axis=1), y_pred))
print "-----------------------------------------------------------"
start_time=time.time()
par_sgd_dropout_09,loss_sgd_dropout_09=softmax.sgd_dropout(X_train,y_train,K,start_p,hyper_p,eta=eta,epochs=epochs,batch_size=batch_size,verbose=0,p=0.9)
elapsed_time=time.time()-start_time 
print('SGD Dropout 0.9, time:',elapsed_time)
y_pred=softmax.predict(X_test,par_sgd_dropout_09)
cnf_matrix_dropout_09=confusion_matrix(y_test[:].argmax(axis=1), y_pred)
print(classification_report(y_test[:].argmax(axis=1), y_pred))
print "-----------------------------------------------------------"



np.savez("results/sgd_plants.npz",par_sgd=par_sgd,par_sgd_dropout_05=par_sgd_dropout_05,par_sgd_dropout_01=par_sgd_dropout_01,par_sgd_dropout_09=par_sgd_dropout_09)

plt.figure()
plot_confusion_matrix(cnf_matrix_sgd, classes=np.int32(classes),title='SGD')
plt.savefig('plants_confusion_matrix_sgd.pdf',bbox_inches='tight')
plt.close()

plt.figure()
plot_confusion_matrix(cnf_matrix_dropout_05, classes=np.int32(classes),title='Dropout $p=0.5$')
plt.savefig('plants_confusion_matrix_dropout_05.pdf',bbox_inches='tight')
plt.close()

plt.figure()
plot_confusion_matrix(cnf_matrix_dropout_01, classes=np.int32(classes),title='Dropout $p=0.1$')
plt.savefig('plants_confusion_matrix_dropout_01.pdf',bbox_inches='tight')
plt.close()

plt.figure()
plot_confusion_matrix(cnf_matrix_dropout_09, classes=np.int32(classes),title='Dropout $p=0.9$')
plt.savefig('plants_confusion_matrix_dropout_09.pdf',bbox_inches='tight')
plt.close()

sns.set()
plt.figure()
plt.plot(range(epochs),loss_sgd,'-',label='SGD')
plt.plot(range(epochs),loss_sgd_dropout_09,':',label='Dropout $p=0.9$')
plt.plot(range(epochs),loss_sgd_dropout_05,'.-',label='Dropout $p=0.5$')
plt.plot(range(epochs),loss_sgd_dropout_01,'--',label='Dropout $p=0.1$')
#plt.title('Training loss')
plt.ylabel('log-loss')
plt.xlabel('epochs')
plt.legend(loc='best')
plt.savefig('plants_fine_tuning.pdf',bbox_inches='tight')
plt.close() """


