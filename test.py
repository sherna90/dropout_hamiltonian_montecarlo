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
hyper_p={'alpha':alpha}

n_samples = [100, 1000, 5000, 10000]
#batch_size_list = [50, 100, 200, 300, 500]
batch_size_list = [50]
D_list = [ 10, 50, 100, 500, 1000, 5000, 10000, 20000]

df = pd.DataFrame()

for j in range(len(D_list)):
    D=D_list[j]
    start_p={'weights':np.zeros(D),'bias':np.zeros(1)}
    centers = [np.random.random_integers(0,10,D),np.random.random_integers(0,10,D)]

    for i in range(len(n_samples)):

        X, y = make_blobs(n_samples=n_samples[i], centers=centers, cluster_std=10,random_state=40)
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


        import hamiltonian.logisticcpu as logistic
        LOG=logistic.LOGISTIC()

        start_cpu = t.time()
        par,loss=LOG.sgd(X_train.copy(), y_train.copy(), start_p.copy(), hyper_p,eta=1e-5,epochs=1e3,batch_size=batch_size_list[0],verbose=False)
        end_cpu = t.time() - start_cpu
        print("D:{} - n_sample:{} => {}".format(D_list[j], n_samples[i], end_cpu))
        #LOG.stats()
        
        df['cpu_D:{}n_sample:{}'.format(D_list[j], n_samples[i])] = end_cpu

        #########

        #if gpu:
        import hamiltonian.logisticgpu as logistic
        LOG=logistic.LOGISTIC()

        start_gpu = t.time()
        par,loss=LOG.sgd(X_train.copy(), y_train.copy(), start_p.copy(), hyper_p,eta=1e-5,epochs=1e3,batch_size=batch_size_list[0],verbose=False)
        end_gpu = t.time() - start_gpu
        print("D:{} - n_sample:{} => {}".format(D_list[j], n_samples[i], end_gpu))
        #LOG.stats()

        df['gpu_D:{}n_sample:{}'.format(D_list[j], n_samples[i])] = end_gpu

        if end_gpu < end_cpu:
            print("Gano la gpu con {} frente a {}".format(end_gpu, end_cpu))

df.to_csv('times2.csv', sep='\t', encoding='utf-8')