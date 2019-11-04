import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

import sys 
sys.path.append('./')
import hamiltonian.utils as utils
import hamiltonian.models.gpu.logistic as base_model
import hamiltonian.inference.gpu.sgd as inference

X_train = np.loadtxt('data/X_train.csv',delimiter=',')
y_train = np.loadtxt('data/y_train.csv',delimiter=',')


D=X_train.shape[1]

epochs = 1e3
eta=1e-4
batch_size=500
alpha=1/10.
dropout_rate=1.0

start_p={'weights':np.random.random((D,1)),
        'bias':np.random.random(1)}
hyper_p={'alpha':alpha}

model=base_model.logistic(hyper_p)
optim=inference.sgd(model,start_p,step_size=eta)
par,loss=optim.fit(epochs=epochs,batch_size=batch_size,gamma=0.99,X_train=X_train,y_train=y_train,verbose=True)



