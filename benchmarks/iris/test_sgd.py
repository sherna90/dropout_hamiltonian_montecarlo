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
import hamiltonian.models.cpu.logistic as base_model
import hamiltonian.inference.cpu.sgd as inference

iris = datasets.load_iris()
data = iris.data  
labels = iris.target
classes=np.logical_or(labels==0 ,labels==1)
# species == ('setosa', 'versicolor')
# #['petal_length', 'petal_width'] 
X_train, y_train = iris.data[classes,0:2], iris.target[classes]
#X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)

D=X_train.shape[1]
N=X_train.shape[0]



epochs = 4e4
eta=1e-5
batch_size=50
alpha=1/100.
dropout_rate=1.0

start_p={'weights':2*np.random.random((D,1)),'bias':2*np.random.random(1)}
hyper_p={'alpha':alpha}

model=base_model.logistic(hyper_p)
optim=inference.sgd(model,start_p,step_size=eta)
par,loss=optim.fit(epochs=epochs,batch_size=N,gamma=0.9,X_train=X_train,y_train=y_train,verbose=True)
y_pred=model.predict(par,X_train,batchsize=N)
print(par)
print(classification_report(y_train, y_pred))
print(confusion_matrix(y_train, y_pred))

fig, ax = plt.subplots(figsize=(10,7))
ax.plot(loss)
ax.set_xlabel("Epochs")
ax.set_ylabel("Negative Log-loss")
plt.show()
