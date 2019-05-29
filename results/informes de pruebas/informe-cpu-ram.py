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

from multiprocessing import Pool,cpu_count
import hamiltonian.hmccpu as hmc

alpha=1./4.
hyper_p={'alpha':alpha}

ncores=cpu_count()
#backend = 'simulated_data'
backend = None
niter = 1e3
burnin = 1e2

n_samples = [100, 1000, 5000, 10000]
D_list = [10, 50]

import hamiltonian.logisticcpu as logistic
import time

df = pd.DataFrame(columns=['D', 'n', 'Time'])
cont = 0

LOG=logistic.LOGISTIC()

for j in range(len(D_list)):
    D=D_list[j]
    start_p={'weights':np.zeros(D),'bias':np.zeros(1)}
    centers = [np.random.random_integers(0,10,D),np.random.random_integers(0,10,D)]

    for i in range(len(n_samples)):
        X, y = make_blobs(n_samples=n_samples[i], centers=centers, cluster_std=10,random_state=40)
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        #par,loss=LOG.sgd(X_train, y_train, start_p, hyper_p, eta=1e-5,epochs=1e4,batch_size=50,verbose=False)

        mcmc=hmc.HMC(X_train,y_train,LOG.loss, LOG.grad, start_p, hyper_p, path_length=1,verbose=0)
        
        start = time.time()
        posterior_sample,logp_samples=mcmc.multicore_sample(niter,burnin,backend=backend, ncores=ncores)
        end = time.time() - start
        print("{} {} {}".format(D_list[j], n_samples[i], end))
        df.loc[cont] = [D_list[j], n_samples[i], end]
        cont += 1

        '''
        par_mean={var:np.mean(posterior_sample[var],axis=0).reshape(start_p[var].shape) for var in posterior_sample.keys()}
        #par_var={var:np.var(posterior_sample[var],axis=0).reshape(start_p[var].shape) for var in posterior_sample.keys()}

        y_pred=LOG.predict(X_test,par_mean)

        
        print("D:{} - n_sample:{}: {}".format(D_list[j], n_samples[i], end))

        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        '''
df.to_csv('informe-cpu-ram.csv', sep='\t')
