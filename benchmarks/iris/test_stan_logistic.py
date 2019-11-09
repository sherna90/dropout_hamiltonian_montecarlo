import numpy as np

from sklearn import datasets
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pystan
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
plt.style.use('seaborn-white')

iris = datasets.load_iris()
data = iris.data  
labels = iris.target
classes=np.logical_or(labels==0 ,labels==1)
# species == ('setosa', 'versicolor')
# ['sepal_length', 'sepal_width']
X_train, y_train = iris.data[classes,0:2], iris.target[classes]
#X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)

D=X_train.shape[1]
N=X_train.shape[0]

#Gaussian-prior stan model
model_code = """
data {
    int<lower=0> N;
    int<lower=0> d;
    matrix[N,d] X;
    int y[N];
    matrix[d,d] Sigma0;
}

transformed data {
    vector[d] mu0;
    for (j in 1:d)
        mu0[j] = 0;
}

parameters {
    vector[d] beta;
    real alpha;
}

transformed parameters {
    real Xbeta[N];
    for (i in 1:N)
        Xbeta[i] = dot_product(row(X, i), beta)+alpha;
}

model {
    alpha ~ normal(0,10);
    beta ~ multi_normal(mu0, Sigma0);
    y ~ bernoulli_logit(Xbeta);
}
"""


data = dict(N=N,X=X_train,y=y_train,d=D,Sigma0=np.identity(D)*10)

algorithm="NUTS" #put NUTS or HMC
iterations=2000 #iterations of algorithm

print (" ------------------------------------------------------------------------------------------------------------")
print ("| Running Gaussian-prior with: ",algorithm ," | Iterations: ",iterations," | N: ",N, " | D:", D)
print (" ------------------------------------------------------------------------------------------------------------")


if True:
    sm = pystan.StanModel(model_code=model_code)
    with open('stan_model.pkl', 'wb') as f:
        pickle.dump(sm, f)
else:
    sm = pickle.load(open('stan_model.pkl', 'rb'))


control = {"stepsize" : 0.1,
           "adapt_engaged" : False}
fit = sm.sampling(data=data, algorithm=algorithm,chains=4)

import pandas as pd

samples=np.concatenate((fit.extract()['beta'],fit.extract()['alpha'].reshape(-1,1)),axis=1)
b_sample = pd.DataFrame(samples,columns=['b0','b1','alpha'])

print(b_sample.describe())
op = sm.optimizing(data=data)
print('MAP beta : {0}, alpha : {1}'.format(op['beta'],op['alpha']))
fig, ax =plt.subplots(1,3)
sns.distplot(b_sample['b0'], ax=ax[0])
sns.distplot(b_sample['b1'], ax=ax[1])
sns.distplot(b_sample['alpha'], ax=ax[2])
plt.show()
#sns.pairplot(b_sample)
#plt.show()