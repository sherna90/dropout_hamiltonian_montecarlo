import warnings
warnings.filterwarnings("ignore")

from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import h5py 
import sys 
sys.path.append('./')
import hamiltonian.utils as utils
import hamiltonian.models.cpu.softmax as base_model
import hamiltonian.inference.cpu.sgd as inference


eta=1e-2
epochs=100
batch_size=250
alpha=1e-2
data_path = './data/'

plants_train=h5py.File(data_path+'train_features_labels.h5','r')
X_train=plants_train['train_features']
y_train=plants_train['train_labels']
plants_test=h5py.File(data_path+'validation_features_labels.h5','r')
X_test=plants_test['validation_features']
y_test=plants_test['validation_labels']

D=X_train.shape[1]
K=y_train.shape[1]
import time

start_p={'weights':np.random.random((D,K)),
        'bias':np.random.random((K))}
hyper_p={'alpha':alpha}

start_time=time.time()
model=base_model.softmax(hyper_p)
optim=inference.sgd(model,start_p,step_size=eta)
par,loss=optim.fit_dropout(epochs=epochs,batch_size=batch_size,p=0.5,gamma=0.9,X_train=X_train,y_train=y_train,verbose=True)
print('SGD, time:',time.time()-start_time)

samples=[]
for i in range(50):
    y_pred=model.predict_stochastic(par,X_test,p=0.5,prob=True)
    samples.append(y_pred)

df_list=[pd.DataFrame(p) for p in samples]
with pd.ExcelWriter('output.xlsx') as writer:
    for i,df in enumerate(df_list):
        df.to_excel(writer, engine='xlsxwriter',sheet_name='sample_{}'.format(i))

y_pred=np.median(samples,axis=0)
cnf_matrix_sgd=confusion_matrix(y_test[:].argmax(axis=1), y_pred.argmax(axis=1))
print(classification_report(y_test[:].argmax(axis=1), y_pred.argmax(axis=1)))
print("-----------------------------------------------------------")
plants_train.close()
plants_test.close()
loss=pd.DataFrame(loss_sgd)
if use_gpu:
    loss.to_csv('loss_sgd_gpu.csv',sep=',',header=False)
    plt.figure()
    plot_confusion_matrix(cnf_matrix_sgd, classes=np.int32(K),title='SGD GPU')
    plt.savefig('plants_confusion_matrix_sgd_gpu.pdf',bbox_inches='tight')
    plt.close()
else:
    loss.to_csv('loss_sgd_cpu.csv',sep=',',header=False)
    plt.figure()
    plot_confusion_matrix(cnf_matrix_sgd, classes=np.int32(K),title='SGD CPU')
    plt.savefig('plants_confusion_matrix_sgd_cpu.pdf',bbox_inches='tight')
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


