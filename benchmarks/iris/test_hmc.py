import warnings
warnings.filterwarnings("ignore")

from sklearn import datasets
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
import sys 
import pandas as pd
import time

sys.path.append("../../") 
import hamiltonian.cpu.softmax as softmax
import hamiltonian.cpu.hmc as sampler
import hamiltonian.utils as utils
import hamiltonian.utils as utils
from scipy import stats


alpha=1/4.
path_length=20
iris = datasets.load_iris()
data = iris.data  
labels = iris.target
classes=np.unique(iris.target)
X, y = iris.data, iris.target
num_classes=len(classes)
K=num_classes
y=utils.one_hot(y,num_classes)
X = (X - X.mean(axis=0)) / X.std(axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,shuffle=True)

D=X_train.shape[1]

start_p={'weights':np.zeros((D,K)),
        'bias':np.zeros((K))}
hyper_p={'alpha':alpha}
model=softmax.SOFTMAX()
mcmc=sampler.hmc(model.log_likelihood, model.grad, start_p,hyper_p, path_length=10,verbose=0)
t0=time.clock()
posterior_sample,logp_samples=mcmc.sample(X_train,y_train,1e3,1e2)
t1=time.clock()
print("Ellapsed Time : ",t1-t0)

post_par={var:np.mean(posterior_sample[var],axis=0).reshape(start_p[var].shape) for var in posterior_sample.keys()}
y_pred=model.predict(X_test,post_par)
print(classification_report(y_test.argmax(axis=1), y_pred))
print(confusion_matrix(y_test.argmax(axis=1), y_pred))

b_cols=columns=['b1', 'b2','b3']


b_sample = pd.DataFrame(posterior_sample['bias'], columns=b_cols)
w_sample = pd.DataFrame(posterior_sample['weights'])

print(b_sample.describe())
print(w_sample.describe())
sns.distplot(b_sample['b1'])
sns.distplot(b_sample['b2'])
sns.distplot(b_sample['b3'])
#sns.pairplot(b_sample)
plt.show()
#plt.hist(logp_samples)
#plt.show()
