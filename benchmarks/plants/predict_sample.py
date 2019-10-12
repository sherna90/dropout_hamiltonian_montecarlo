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
from scipy.special import logsumexp
import pandas as pd
import pickle

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

backend = "sgmcmc_plants_gpu.h5"
model=softmax.SOFTMAX()
theta=h5py.File(backend,'r')

model_predictions=[]
for i in range(theta['weights'].shape[0]):
    post_par={'weights':theta['weights'][i],'bias':theta['bias'][i]}
    y_linear = np.dot(X_test[1,:], post_par['weights']) + post_par['bias']
    lse=logsumexp(y_linear)
    y_hat=y_linear-np.repeat(lse,len(y_linear)).reshape(y_linear.shape)
    model_predictions.append(y_hat)

model_predictions=np.array(model_predictions)
my_df = pd.DataFrame(model_predictions) 
my_df.to_csv('predictions.csv', index = False)  