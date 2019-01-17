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

n_samples=10

npz=np.load("sgd_plants.npz")
par_sgd={}
par_sgd.update({0.5:npz['par_sgd_dropout_05'][()]})
par_sgd.update({0.1:npz['par_sgd_dropout_01'][()]})
par_sgd.update({0.9:npz['par_sgd_dropout_09'][()]})

predictive_accuracy={k:[] for k in par_sgd.keys()}
for k in range(n_samples):
    for p,par in par_sgd.iteritems():
        y_pred=softmax.predict_stochastic(X_test,par,p=p)
        acc=np.sum(y_test[:].argmax(axis=1)==y_pred)/np.float(len(y_pred))
        predictive_accuracy[p].append(acc)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

for p in predictive_accuracy.keys():
    m=np.mean(predictive_accuracy[p])
    v=np.var(predictive_accuracy[p])
    print('MC Dropout {0:01.2f}, mean accuracy : {1:01.2f}, var accuracy : {2:01.2f}'.format(p,m,v))
    plt.hist(predictive_accuracy[p])
    plt.title('MC dropout $p='+str(p)+'$')
    plt.xlabel('accuracy')
    plt.savefig('accuracy_mc_dropout_'+str(p)+'.pdf',bbox_inches='tight')
    plt.close()

plants_test.close()
