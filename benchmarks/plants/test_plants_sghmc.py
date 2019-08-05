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
    import hamiltonian.cpu.sgld as sampler

eta=1e-3
epochs=1
batch_size=50
alpha=1e-2
burnin=1
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
backend = "sgmcmc_plants.h5"
#backend = None
model=softmax.SOFTMAX()
mcmc=sampler.sgld(model, start_p,hyper_p, path_length=1,verbose=1)
start_time=time.time()
posterior_sample,logp_samples=mcmc.sample(X_train,y_train,epochs,burnin,batch_size=batch_size, backend=backend)
elapsed_time=time.time()-start_time 
print("Ellapsed Time : {0:.4f}".format(elapsed_time))

if backend:
    par_mean = mcmc.backend_mean(backend, epochs)

    y_pred_mc=model.predict(X_test,par_mean)

    print(classification_report(y_test[:].argmax(axis=1), y_pred_mc))
    print(confusion_matrix(y_test[:].argmax(axis=1), y_pred_mc))
else:
    post_par={var:np.mean(posterior_sample[var],axis=0).reshape(start_p[var].shape) for var in posterior_sample.keys()}
    #post_par_var={var:np.var(posterior_sample[var],axis=0).reshape(start_p[var].shape) for var in posterior_sample.keys()}
    y_pred=model.predict(X_test,post_par)
    print(classification_report(y_test[:].argmax(axis=1), y_pred))
    print(confusion_matrix(y_test[:].argmax(axis=1), y_pred))


plants_train.close()
plants_test.close()

if use_gpu:
    loss.to_csv('loss_sgd_gpu.csv',sep=',',header=False)
    plt.figure()
    plot_confusion_matrix(cnf_matrix_sgd, classes=np.int32(K),title='SGD GPU')
    plt.savefig('plants_confusion_matrix_sgd_gpu.pdf',bbox_inches='tight')
    plt.close()
else:
    loss.to_csv('loss_sgd_cpu.csv',sep=',',header=False)
    plt.figure()
    plot_confusion_matrix(cnf_matrix_sgd, classes=np.int32(K),title='SGD CPU')
    plt.savefig('plants_confusion_matrix_sgd_cpu.pdf',bbox_inches='tight')
    plt.close()