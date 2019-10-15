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
import hamiltonian.inference.cpu.sgd as inference

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


epochs = 3e4
eta=1e-4
batch_size=50
alpha=1/10.
dropout_rate=1.0

start_p={'weights':np.random.random((D,K)),
        'bias':np.random.random((K))}
hyper_p={'alpha':alpha}

model=base_model.softmax(hyper_p)
optim=inference.sgd(model,start_p,step_size=eta)
par,loss=optim.fit(epochs=epochs,batch_size=30,gamma=0.99,X_train=X_train,y_train=y_train,verbose=True)
y_pred=model.predict(par,X_test)

print(classification_report(y_test.argmax(axis=1), y_pred))
print(confusion_matrix(y_test.argmax(axis=1), y_pred))

fig, ax = plt.subplots(figsize=(10,7))
ax.plot(loss)
ax.set_xlabel("Epochs")
ax.set_ylabel("Negative Log-loss")
plt.show()
