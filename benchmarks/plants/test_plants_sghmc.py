import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import sys
sys.path.append("../../") 
import time
import h5py
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import itertools
from confusion_matrix import *
import hamiltonian.utils as utils


print('GPU/CPU: {}'.format(sys.argv[1]))

if len(sys.argv)>1 and sys.argv[1]=='gpu':
    use_gpu=True
else:
    use_gpu=False

if use_gpu:
    import hamiltonian.gpu.softmax as softmax
    import hamiltonian.gpu.sgld as sampler
else:
    import hamiltonian.cpu.softmax as softmax
    import hamiltonian.cpu.sgld_multicore as sampler

eta=1e-3
epochs=100
batch_size=50
alpha=1e-2
burnin=10
data_path = '/home/sergio/data/PlantVillage-Dataset/balanced_train_test/features/'

plants_train=h5py.File(data_path+'plant_village_train.hdf5','r')
X_train=plants_train['features']
y_train=plants_train['labels']
plants_test=h5py.File(data_path+'plant_village_val.hdf5','r')
X_test=plants_test['features']
y_test=plants_test['labels']


D=X_train.shape[1]
K=y_train.shape[1]
import time

start_p={'weights':np.random.random((D,K)),
        'bias':np.random.random((K))}
hyper_p={'alpha':alpha}
#backend = "sgmcmc_plants.h5"
backend = None
model=softmax.SOFTMAX()
if use_gpu:
    mcmc=sampler.sgld(model, start_p,hyper_p, path_length=1,verbose=1)
    start_time=time.time()
    posterior_sample,loss_sgld=mcmc.sample(X_train,y_train,epochs,burnin,batch_size=batch_size, backend=backend)
    elapsed_time=time.time()-start_time
else:
     mcmc=sampler.sgld_multicore(model, start_p,hyper_p, path_length=1,verbose=1)
     start_time=time.time()
     posterior_sample,loss_sgld=mcmc.multicore_sample(X_train,y_train,epochs,burnin,batch_size=batch_size, backend=backend,ncores=4)
     elapsed_time=time.time()-start_time

print("Ellapsed Time : {0:.4f}".format(elapsed_time))

if backend:
    backend_name = posterior_sample
    par_mean = mcmc.backend_mean(backend_name, epochs)
    y_pred=model.predict(X_test,par_mean)
    cnf_matrix_sgld=confusion_matrix(y_test[:].argmax(axis=1), y_pred)
    print(classification_report(y_test[:].argmax(axis=1), y_pred))
    print(cnf_matrix_sgld)
else:
    post_par={var:np.mean(posterior_sample[var],axis=0).reshape(start_p[var].shape) for var in posterior_sample.keys()}
    y_pred=model.predict(X_test,post_par)
    cnf_matrix_sgld=confusion_matrix(y_test[:].argmax(axis=1), y_pred)
    print(classification_report(y_test[:].argmax(axis=1), y_pred))
    print(cnf_matrix_sgld)


plants_train.close()
plants_test.close()

loss=pd.DataFrame(loss_sgld)
if use_gpu:
    loss.to_csv('loss_sgld_gpu.csv',sep=',',header=False)
    plt.figure()
    plot_confusion_matrix(cnf_matrix_sgld, classes=np.int32(K),title='SGLD GPU')
    plt.savefig('plants_confusion_matrix_sgld_gpu.pdf',bbox_inches='tight')
    plt.close()
else:
    loss.to_csv('loss_sgld_cpu.csv',sep=',',header=False)
    plt.figure()
    plot_confusion_matrix(cnf_matrix_sgld, classes=np.int32(K),title='SGLD CPU')
    plt.savefig('plants_confusion_matrix_sgld_cpu.pdf',bbox_inches='tight')
    plt.close()