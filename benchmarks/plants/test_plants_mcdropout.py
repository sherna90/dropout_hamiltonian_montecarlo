import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import sys
sys.path.append('./')
import time
import h5py

use_gpu=False
if use_gpu:
    import hamiltonian.softmax_gpu as softmax
else:
    import hamiltonian.softmax as softmax

plants_test=h5py.File('data/validation_features_labels.h5','r')
X_test=plants_test['validation_features']
y_test=plants_test['validation_labels']
D=X_test.shape[1]
K=y_test.shape[1]
n_samples=1000

npz=np.load("results/sgd_plants.npz")
par_sgd={}
par_sgd.update({0.5:npz['par_sgd_dropout_05'][()]})
par_sgd.update({0.1:npz['par_sgd_dropout_01'][()]})
par_sgd.update({0.9:npz['par_sgd_dropout_09'][()]})

predictive_accuracy={k:[] for k in par_sgd.keys()}
ellapsed_time={}
for p,par in par_sgd.iteritems():
    t0=time.clock()
    for k in range(n_samples):
        y_pred=softmax.predict_stochastic(X_test,par,p=p)
        acc=np.sum(y_test[:].argmax(axis=1)==y_pred)/np.float(len(y_pred))
        predictive_accuracy[p].append(acc)
    t1=time.clock()
    eps=t1-t0
    ellapsed_time[p]=eps
    print("MC dropout Ellapsed Time : ",eps)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

posterior_sample=h5py.File('results/sgmcmc_plants.h5','r')
mcmc_samples=posterior_sample['weights'].shape[0]
hyper_p={'alpha':1e-2}
predictive_accuracy.update({'sgld':[]})

t0=time.clock()
for i in range(1,mcmc_samples):
    par={'weights':posterior_sample['weights'][i,:],'bias':posterior_sample['bias'][i,:]}
    y_pred=softmax.predict(X_test,par)
    acc=np.sum(y_test[:].argmax(axis=1)==y_pred)/np.float(len(y_pred))
    predictive_accuracy['sgld'].append(acc)
t1=time.clock()
eps=t1-t0
ellapsed_time['sgld']=eps
print("SGLD Ellapsed Time : ",eps)

np.savez("results/results_bayesian_plants.npz",predictive_accuracy=predictive_accuracy,ellapsed_time=ellapsed_time)

if False:
    for p in predictive_accuracy.keys():
        m=np.mean(predictive_accuracy[p])
        v=np.var(predictive_accuracy[p])
        print('Method :'+str(p)+', mean : '+str(m)+', var:'+str(v))
        plt.hist(predictive_accuracy[p])
        plt.xlabel('accuracy')
        plt.savefig('accuracy_bayesian_'+str(p)+'.pdf',bbox_inches='tight')
        #plt.close()


plants_test.close()