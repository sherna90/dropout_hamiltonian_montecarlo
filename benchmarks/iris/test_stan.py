import numpy as np

from sklearn import datasets
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pystan
import sys
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
sys.path.append("./") 

scaler = StandardScaler()
iris = datasets.load_iris()
data = iris.data 
labels = iris.target + 1 
classes=np.unique(iris.target)
X, y = iris.data, iris.target + 1 
X=scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#Gaussian-prior stan model
model_code = """
data {
  int<lower=0> N;
  int<lower=0> D;
  int<lower=2> K;
  row_vector[D] X[N];
  int<lower=1,upper=K> y[N];
}
parameters {
  matrix[D,K] weights;
  vector[K] bias;
}
transformed parameters {
  simplex[K] theta[N];
  for (n in 1:N)
    theta[n] = softmax(to_vector(X[n]*weights)+bias);
}
model {
  for (i in 1:D){
    for (j in 1:K){
      weights[i,j] ~ normal(0, 2);
    }
  }
  for (k in 1:K){
      bias[k] ~ normal(0, 2);
  }
  for (n in 1:N)
    y[n] ~ categorical(theta[n]);
}
"""

N, D = X_train.shape 
K=len(classes)
data = dict(N=N, D=D,K=K, X=X_train, y=y_train)

algorithm="HMC" #put NUTS or HMC
iterations=2000 #iterations of algorithm

print " ------------------------------------------------------------------------------------------------------------"
print "| Running Gaussian-prior with: ",algorithm ," | Iterations: ",iterations," | N: ",N, " | D:", D, " | K:", K
print " ------------------------------------------------------------------------------------------------------------"

"""sm = pystan.StanModel(model_code=model_code)
"op = sm.optimizing(data=data)
"print op"""

fit = pystan.stan(model_code=model_code, data=data, seed=5, iter=iterations, algorithm=algorithm)

post_par={'weights':np.mean(fit.extract()['weights'], axis=0),'bias':np.mean(fit.extract()['bias'], axis=0)}

#y_pred=softmax.predict(X_test,post_par)
#print(classification_report(y_test, y_pred))
#print(confusion_matrix(y_test, y_pred))

import pandas as pd

b_cols=columns=['b1', 'b2','b3']
b_sample = pd.DataFrame(fit.extract()['bias'], columns=b_cols)
print(b_sample.describe())
#w_sample = pd.DataFrame(fit.extract()['weights'].reshape(-1))
sns.pairplot(b_sample)
plt.show()