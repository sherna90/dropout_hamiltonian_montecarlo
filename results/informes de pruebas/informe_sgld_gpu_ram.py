from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

import sys
sys.path.append("../../") 
import numpy as np
import time

import hamiltonian.utils as utils
import hamiltonian.gpu.softmax as softmax

import hamiltonian.gpu.sgld as sampler

alpha=1./4.
hyper_p={'alpha':alpha}


niter = 1e4
burnin=1e2
n_samples = [1000, 10000, 50000, 100000]
D_list = [10, 50,100]
num_classes = 10

df = pd.DataFrame(columns=['D', 'n', 'Time'])

cont = 0
SOFT=softmax.SOFTMAX()

for j in range(len(D_list)):
        D=D_list[j]
        for i in range(len(n_samples)):
                centers = [np.random.random_integers(0,10,D) for i in range(num_classes)]
                X, y = make_blobs(n_samples=n_samples[i], centers=centers, cluster_std=1,random_state=40)
                y=utils.one_hot(y,num_classes)
                X = (X - X.mean(axis=0)) / X.std(axis=0)
                X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
                alpha=1./4.
                start_p={'weights':np.zeros((D,num_classes)),'bias':np.zeros((num_classes))}
                hyper_p={'alpha':alpha}
                mcmc=sampler.sgld(SOFT.loss, SOFT.grad, start_p,hyper_p, path_length=1,verbose=0)
                start=time.time()
                posterior_sample,logp_samples=mcmc.sample(X_train,y_train,niter,burnin,batch_size=50, backend=None)
                end = time.time() - start
                print("{} {} {}".format(D_list[j], n_samples[i], end))
                df.loc[cont] = [D_list[j], n_samples[i], end]
                cont += 1

df.to_csv('informe-gpu-ram.csv', sep='\t')
