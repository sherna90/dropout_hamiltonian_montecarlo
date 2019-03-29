from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np

D=2
centers = [np.random.random_integers(0,10,D),np.random.random_integers(0,10,D)]
X, y = make_blobs(n_samples=100, centers=centers, cluster_std=10,random_state=40)
X = (X - X.mean(axis=0)) / X.std(axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#############################
 
import sys
sys.path.append("../") 
import numpy as np
import time

gpu = False
alpha=1./4.

if gpu:
    import hamiltonian.logisticgpu as logistic
    LOG=logistic.LOGISTIC(X_train, y_train, alpha, D)

    star = time.time()
    par,loss=LOG.sgd(eta=1e-5,epochs=1e4,batch_size=50,verbose=True)
    print time.time() - star
else:
    import hamiltonian.logisticcpu as logistic
    LOG=logistic.LOGISTIC(X_train, y_train, alpha, D)

    star = time.time()
    par,loss=LOG.sgd(eta=1e-5,epochs=1e4,batch_size=50,verbose=True)
    print time.time() - star

y_pred=LOG.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))