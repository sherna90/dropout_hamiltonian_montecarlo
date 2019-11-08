import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import sys 
sys.path.append('./')
import hamiltonian.utils as utils
import hamiltonian.models.cpu.logistic as base_model
import hamiltonian.inference.cpu.hmc as inference

X = np.loadtxt('data/X_train.csv',delimiter=',')
y = np.loadtxt('data/y_train.csv',delimiter=',')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

D=X_train.shape[1]
N=X_train.shape[0]


epochs = 1e4
eta=0.1
batch_size=500
alpha=1/10.
dropout_rate=1.0
burnin=1e3

start_p={'weights':np.random.random((D,1)),
        'bias':np.random.random(1)}
hyper_p={'alpha':alpha}

model=base_model.logistic(hyper_p)
hmc=inference.hmc(model,start_p,path_length=10,step_size=eta)
samples,loss,positions,momentums=hmc.sample(niter=epochs,burnin=burnin,rng=None,X_train=X_train,y_train=y_train)
post_par={var:np.median(samples[var],axis=0) for var in samples.keys()}
y_pred=model.predict(post_par,X_train)

print(classification_report(y_train, y_pred))
print(confusion_matrix(y_train, y_pred))

import pandas as  pd
import matplotlib.pyplot as plt
output=pd.DataFrame(np.squeeze(samples['weights'],axis=2))
output.boxplot()
plt.show()
