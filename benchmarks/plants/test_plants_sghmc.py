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

eta=1e-2
epochs=100
burnin=100
batch_size=32
alpha=1e-2
data_path = './data/'

train_file='plant_village_train.hdf5'
test_file='plant_village_val.hdf5'

plants_train=h5py.File(data_path+train_file,'r')
X_train=plants_train['features']
y_train=plants_train['labels']
plants_test=h5py.File(data_path+test_file,'r')
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

def train_model():
    sampler=inference.sgld(model,start_p,path_length=eta,step_size=eta)
    samples,loss=sampler.sample(epochs=epochs,burnin=burnin,batch_size=batch_size,X_train=X_train,y_train=y_train,verbose=True)
    post_par={var:np.median(samples[var],axis=0) for var in samples.keys()}
    y_pred=model.predict(post_par,X_test,prob=True)
    print('SGHMC, time:',time.time()-start_time)
    loss=pd.DataFrame(loss)
    loss.to_csv('loss.csv',sep=',',header=False)
    with open('model.pkl','wb') as handler:
        pickle.dump(samples,handler)

def test_model():
    with open('model.pkl','rb') as handler:
        samples=pickle.load(handler)
    predict_samples=[]
    for i in range(epochs):
        print('prediction : {0}'.format(i))
        par={var:samples[var][i] for var in samples.keys()}
        y_pred=model.predict(par,X_test,prob=True)
        y_pred=y_pred.reshape(-1, y_pred.shape[-1])
        predict_samples.append(y_pred)
    with h5py.File('output.hdf5', 'w') as f:
        f["predict_samples"] = np.asarray(predict_samples)
    print(classification_report(y_test[:].argmax(axis=1), y_pred.argmax(axis=1)))
    print("-----------------------------------------------------------")


train_model()
test_model()

plants_train.close()
plants_test.close()
