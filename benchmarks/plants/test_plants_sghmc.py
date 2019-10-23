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
import hamiltonian.models.gpu.softmax as base_model
import hamiltonian.inference.gpu.sgld as inference
import pickle

eta=1e-1
epochs=100
burnin=10
batch_size=30
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
sampler=inference.sgld(model,start_p,path_length=eta,step_size=eta)
samples,loss=sampler.sample(epochs=epochs,burnin=burnin,batch_size=batch_size,X_train=X_train,y_train=y_train,verbose=True)
post_par={var:np.median(samples[var],axis=0) for var in samples.keys()}
y_pred=model.predict(post_par,X_test,prob=True)
print('SGHMC, time:',time.time()-start_time)

with open('sgld_model.pkl','wb') as handler:
    pickle.dump(samples,handler)


with open('sgld_loss.pkl','wb') as handler:
    pickle.dump(loss,handler)

predict_samples=[]
for i in range(epochs):
    par={var:samples[var][i] for var in samples.keys()}
    y_pred=model.predict_stochastic(par,X_test,p=0.5,prob=True)
    predict_samples.append(y_pred)

df_list=[pd.DataFrame(p) for p in predict_samples]
with pd.ExcelWriter('output.xlsx') as writer:
    for i,df in enumerate(df_list):
        df.to_excel(writer, engine='xlsxwriter',sheet_name='sample_{}'.format(i))

print(classification_report(y_test[:].argmax(axis=1), y_pred.argmax(axis=1)))
print("-----------------------------------------------------------")
plants_train.close()
plants_test.close()
