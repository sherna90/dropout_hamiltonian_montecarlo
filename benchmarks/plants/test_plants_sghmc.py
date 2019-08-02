import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import sys
sys.path.append("../../") 
import time
import h5py
import pandas as pd
import hamiltonian.utils as utils
use_gpu=True
if use_gpu:
    import hamiltonian.gpu.softmax as softmax
    import hamiltonian.gpu.sgld as sampler
else:
    import hamiltonian.cpu.softmax as softmax
    import hamiltonian.cpu.sgld as sampler

import matplotlib.pyplot as plt 
import seaborn as sns
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Spectral):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(classes)
    plt.xticks(tick_marks, tick_marks, rotation=45,fontsize=8)
    plt.yticks(tick_marks, tick_marks,fontsize=8)
    fmt = '.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

eta=1e-3
niter=100
batch_size=50
alpha=1e-2
burnin=1e3
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

start_p={'weights':np.zeros((D,K)),
        'bias':np.zeros((K))}
hyper_p={'alpha':alpha}
backend = "sgmcmc_plants.h5"
backend = None
model=softmax.SOFTMAX()
mcmc=sampler.sgld(model.loss, model.grad, start_p,hyper_p, path_length=1,verbose=0)
start=time.time()
posterior_sample,logp_samples=mcmc.sample(X_train,y_train,niter,burnin,batch_size=batch_size, backend=backend)
t1=time.clock()
print("Ellapsed Time : ",t1-t0)

post_par={var:np.mean(posterior_sample[var],axis=0) for var in posterior_sample.keys()}
y_pred=softmax.predict(X_test,post_par)
print(classification_report(y_test[:].argmax(axis=1), y_pred))
print(confusion_matrix(y_test[:].argmax(axis=1), y_pred))

plants_train.close()
plants_test.close()
loss=pd.DataFrame(logp_samples)
loss.to_csv('loss_sgld_gpu.csv',sep=',',header=False)
plt.figure()
plot_confusion_matrix(cnf_matrix_sgd, classes=np.int32(K),title='SGLD GPU')
plt.savefig('plants_confusion_matrix_sgld_gpu.pdf',bbox_inches='tight')
plt.close()