import warnings
warnings.filterwarnings("ignore")

from sklearn import datasets
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import sys 
sys.path.append('./')
import hamiltonian.utils as utils
import hamiltonian.models.cpu.softmax as base_model
import hamiltonian.inference.cpu.sghmc as inference

iris = datasets.load_iris()
data = iris.data  
labels = iris.target
classes=np.unique(iris.target)
X, y = iris.data, iris.target
X = (X - X.mean(axis=0)) / X.std(axis=0)
num_classes=len(classes)
y=utils.one_hot(y,num_classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,shuffle=True)

D=X_train.shape[1]
K=len(classes)


epochs = 1e4
burnin=1e3
eta=1e-2
batch_size=30
alpha=1/100.

start_p={'weights':np.random.random((D,K)),
        'bias':np.random.random((K))}
hyper_p={'alpha':alpha}

model=base_model.softmax(hyper_p)
sampler=inference.sghmc(model,start_p,path_length=1,step_size=eta)
samples,loss=sampler.sample(epochs=epochs,burnin=burnin,batch_size=30,gamma=0.9,X_train=X_train,y_train=y_train)
post_par={var:np.median(samples[var],axis=0) for var in samples.keys()}
y_pred=model.predict(post_par,X_test)

print(classification_report(y_test.argmax(axis=1), y_pred))
print(confusion_matrix(y_test.argmax(axis=1), y_pred))

import pandas as pd
b_data=pd.DataFrame(samples['bias'])
print(b_data.describe())

fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(10,7))
ax[0].plot(loss)
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Log-loss")
ax[1].hist(samples['bias'][:,0])
ax[1].hist(samples['bias'][:,1])
ax[1].hist(samples['bias'][:,2])
ax[1].set_xlabel("Bias")
ax[1].set_ylabel("Freq")
plt.show()
