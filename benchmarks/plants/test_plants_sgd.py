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
import hamiltonian.inference.gpu.sgd as inference
import pickle

eta=1e-5
epochs=100
batch_size=250
alpha=1./100.
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

start_p={'weights':np.zeros((D,K)),
        'bias':np.zeros((K))}
hyper_p={'alpha':alpha}

start_time=time.time()
model=base_model.softmax(hyper_p)

def train_model():
        optim=inference.sgd(model,start_p,step_size=eta)
        par,loss=optim.fit(epochs=epochs,batch_size=batch_size,gamma=0.9,X_train=X_train,y_train=y_train,verbose=True)
        print('SGD, time:',time.time()-start_time)
        loss=pd.DataFrame(loss)
        with open('model.pkl','wb') as handler:
                pickle.dump(par,handler)

def test_model():
        with open('model.pkl','rb') as handler:
                par=pickle.load(handler)
        y_pred=model.predict(par,X_test[:],prob=True,batchsize=32)
        y_pred=y_pred.reshape(-1, y_pred.shape[-1])
        print(classification_report(y_test[:].argmax(axis=1), y_pred.argmax(axis=1)))
        print("-----------------------------------------------------------")

train_model()
test_model()

plants_train.close()
plants_test.close()
