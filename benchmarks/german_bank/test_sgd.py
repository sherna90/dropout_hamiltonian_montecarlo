import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import sys
import time
import h5py
sys.path.append("./")
import hamiltonian.logistic as logistic
import hamiltonian.utils as utils
import numpy as np
import pandas as pd

alpha=1./1.
data_path = 'data/'

X=pd.read_csv('data/german_data.csv',header=None)
y=pd.read_csv('data/german_labels.csv',header=None)
X=X.values
y=y.values
X = (X - X.mean(axis=0)) / X.std(axis=0)
y=y.reshape(-1)
D=X.shape[1]
alpha=1.
start_p={'weights':np.zeros(D),'bias':np.zeros(1)}
hyper_p={'alpha':alpha}

par,loss=logistic.sgd(X,y,start_p,hyper_p,eta=1e-5,epochs=1e4,batch_size=100,verbose=True)
print par['weights']
print par['bias']
