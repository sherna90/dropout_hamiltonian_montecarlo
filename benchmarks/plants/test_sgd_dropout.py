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


eta=1e-3
epochs=100
batch_size=250
alpha=1e-2
data_path = './data/'

plants_train=h5py.File(data_path+'plant_village_train.hdf5','r')
X_train=plants_train['features']
y_train=plants_train['labels']
plants_test=h5py.File(data_path+'plant_village_val.hdf5','r')
X_test=plants_test['features']
y_test=plants_test['labels']

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
loss=pd.DataFrame(loss)
loss.to_csv('loss_sgd_cpu.csv',sep=',',header=False)
plt.figure()
plt.close()
