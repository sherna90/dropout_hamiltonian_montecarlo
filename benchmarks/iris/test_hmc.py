import warnings
warnings.filterwarnings("ignore")

from sklearn import datasets
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import sys 
import pandas as pd

sys.path.append("./") 
use_gpu=False
if use_gpu:
    import hamiltonian.softmax_gpu as softmax
else:
    import hamiltonian.softmax as softmax

import hamiltonian.hmc as hmc

epochs = 100
eta=1e-2
batch_size=10
alpha=1./100.
scaler = StandardScaler()

iris = datasets.load_iris()
data = iris.data  
labels = iris.target
classes=np.unique(iris.target)
X, y = iris.data, iris.target
X=scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


D=X_train.shape[1]
num_classes=len(classes)
start_p={'weights':np.random.randn(D,num_classes),'bias':np.random.randn(num_classes),'alpha':alpha}
par,loss=softmax.sgd(X_train,y_train,num_classes,start_p,eta=eta,epochs=100,batch_size=20,scale=False,transform=True,verbose=0)
mcmc=hmc.HMC(X_train,y_train,softmax.loss, softmax.grad, par, n_steps=100,scale=False,transform=True,verbose=1)
b_sample,w_sample=mcmc.sample(1e4,1e3)
post_par=start_p={'weights':np.mean(w_sample,axis=0).reshape(start_p['weights'].shape),
    'bias':np.mean(b_sample,axis=0).reshape(start_p['bias'].shape),'alpha':alpha}
y_pred=softmax.predict(X_test,post_par,False)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

b_cols=columns=['b1', 'b2','b3']
b_sample = pd.DataFrame(b_sample, columns=b_cols)
print "mean bias : ",b_sample.mean()
print "var bias : ",b_sample.var()
for col in b_cols:
    sns.kdeplot(b_sample[col], shade=True)
plt.show()