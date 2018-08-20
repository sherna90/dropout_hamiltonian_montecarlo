import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pystan

scaler = StandardScaler()
iris = datasets.load_iris()
data = iris.data  
labels = iris.target+1
classes=np.unique(iris.target)
X, y = iris.data, iris.target
X=scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#Gaussian-prior stan model
model_code = """
data {
  int<lower=0> N;
  int<lower=0> D;
  int<lower=2> K;
  vector[D] X[N];
  int y[N];
}
parameters {
  matrix[K,D] weights;
  vector[K] bias;
}
transformed parameters {
  vector[K] theta;
}
model {
  for (i in 1:D){
    bias[i] ~ normal(0, 100);
    for (j in 1:K){
      weights[i,j] ~ normal(0, 100);
    }
  }
  for (n in 1:N)
    y[n] ~ categorical(softmax(weights*X[n]+bias));
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

fit = pystan.stan(model_code=model_code, data=data, seed=5, iter=iterations, algorithm=algorithm)


print "FIT MODEL:",fit
beta = np.mean(fit.extract()['weights'], axis=0)
ypred = np.dot(X_test, beta)



print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

ax1.violinplot(fit.extract()['weights'], points=80, vert=False, widths=0.7, showmeans=True, showextrema=True, showmedians=True)
ax1.set_title('HMC')

