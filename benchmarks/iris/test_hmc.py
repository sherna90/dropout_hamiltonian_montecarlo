import warnings
warnings.filterwarnings("ignore")

from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

import sys 
sys.path.append('./')
import hamiltonian.utils as utils
import hamiltonian.models.gpu.logistic as base_model
import hamiltonian.inference.gpu.hmc as sampler

from importlib import reload

iris = datasets.load_iris()
data = iris.data  
labels = iris.target
classes=np.logical_or(labels==0 ,labels==1)
# species == ('setosa', 'versicolor')
# #['petal_length', 'petal_width'] 
X_train, y_train = iris.data[classes,0:2], iris.target[classes]
#X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)

D=X_train.shape[1]
N=X_train.shape[0]


epochs = 2e3
eta=0.05
batch_size=500
alpha=1/10.
dropout_rate=1.0
burnin=1e3

start_p={'weights':2*np.random.random((D,1)),'bias':np.random.random(1)}
hyper_p={'alpha':alpha}

model=base_model.logistic(hyper_p)
hmc=sampler.hmc(model,start_p,path_length=1,step_size=eta)
samples,loss,positions,momentums=hmc.sample(epochs,burnin,None,X_train=X_train,y_train=y_train)

weights=np.squeeze(samples['weights'],axis=2)
biases=samples['bias']
b_sample = pd.DataFrame(np.concatenate((weights,biases),axis=1),columns=['b0','b1','alpha'])
print(b_sample.describe())
fig, ax =plt.subplots(1,3)
sns.distplot(b_sample['b0'], ax=ax[0])
sns.distplot(b_sample['b1'], ax=ax[1])
sns.distplot(b_sample['alpha'], ax=ax[2])
plt.show()