import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

import sys 
sys.path.append('./')
import hamiltonian.utils as utils
import hamiltonian.models.gpu.logistic as base_model
import hamiltonian.inference.gpu.sgld as inference

X_train = np.loadtxt('data/X_train.csv',delimiter=',')
y_train = np.loadtxt('data/y_train.csv',delimiter=',')


D=X_train.shape[1]
N=X_train.shape[0]


epochs = 1e4
eta=4e-7
batch_size=500
alpha=1/10.
dropout_rate=1.0
burnin=1e3

start_p={'weights':np.random.random((D,1)),
        'bias':np.random.random(1)}
hyper_p={'alpha':alpha}

model=base_model.logistic(hyper_p)
sampler=inference.sgld(model,start_p,path_length=1,step_size=eta)
samples,loss=sampler.sample(epochs=epochs,burnin=burnin,batch_size=batch_size,gamma=0.9,X_train=X_train,y_train=y_train)
post_par={var:np.median(samples[var],axis=0) for var in samples.keys()}
y_pred=model.predict(post_par,X_train)

print(classification_report(y_train, y_pred))
print(confusion_matrix(y_train, y_pred))

import pandas as  pd
import matplotlib.pyplot as plt
output=pd.DataFrame(np.squeeze(samples['weights'],axis=2))
output.boxplot()
plt.show()
