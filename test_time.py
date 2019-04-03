from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np

n_list = [100, 1000]

import sys
sys.path.append("../") 
import time

alpha=1./4.
D=2
centers = [[-5, 0],  [5, -1]]

for i in range(len(n_list)):
    start_p={'weights':np.zeros(D),'bias':np.zeros(1)}
    hyper_p={'alpha':alpha}
    X, y = make_blobs(n_samples=n_list[i], centers=centers, cluster_std=10,random_state=40)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    #CPU#
    import hamiltonian.logisticcpu as logistic
    LOG=logistic.LOGISTIC(X_train, y_train, alpha, D)

    star = time.time()
    par,loss=LOG.sgd(eta=1e-5,epochs=1e4,batch_size=50,verbose=True)
    print "CPU - n: {} => {}\n".format(n_list[i],time.time() - star)

    #GPU#

    import hamiltonian.logisticgpu as logistic
    LOG=logistic.LOGISTIC(X_train, y_train, alpha, D)

    star = time.time()
    par,loss=LOG.sgd(eta=1e-5,epochs=1e4,batch_size=50,verbose=True)
    print "GPU - n: {} => {}\n".format(n_list[i],time.time() - star)

