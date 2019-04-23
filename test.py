from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

import sys
sys.path.append("../") 
import numpy as np
import time as t

alpha=1./4.

batch_size_list = [50, 100, 200, 300, 500]
D_list = [50000, 75000, 100000]

df = pd.DataFrame()

for j in range(len(D_list)):
    D=D_list[j]
    centers = [np.random.random_integers(0,10,D),np.random.random_integers(0,10,D)]
    X, y = make_blobs(n_samples=2000, centers=centers, cluster_std=10,random_state=40)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    for i in range(len(batch_size_list)):
        print "D:{} - Batchsize:{}".format(D_list[j], batch_size_list[i])
        #if gpu:
        import hamiltonian.logisticgpu as logistic
        LOG=logistic.LOGISTIC(X_train, y_train, alpha, D)

        par,loss=LOG.sgd(eta=1e-5,epochs=1e3,batch_size=batch_size_list[i],verbose=False)
        #LOG.stats()

        df['gpu_D:{}_Batchsize:{}'.format(D_list[j], batch_size_list[i])] = LOG.stats()

        #else:
        import hamiltonian.logisticcpu as logistic
        LOG=logistic.LOGISTIC(X_train, y_train, alpha, D)

        par,loss=LOG.sgd(eta=1e-5,epochs=1e3,batch_size=batch_size_list[i],verbose=False)
        #LOG.stats()
        
        df['cpu_D:{}_Batchsize:{}'.format(D_list[j], batch_size_list[i])] = LOG.stats()

df.to_csv('times2.csv', sep='\t', encoding='utf-8')