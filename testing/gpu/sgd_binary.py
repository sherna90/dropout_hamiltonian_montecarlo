from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import sys
sys.path.append("../../") 
import hamiltonian.utils as utils
import hamiltonian.gpu.logistic as logistic

D=2
centers = [[-5, 0],  [5, -1]]
X, y = make_blobs(n_samples=100, centers=centers, cluster_std=1,random_state=40)
X = (X - X.mean(axis=0)) / X.std(axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

alpha=1./4.
start_p={'weights':np.zeros(D),'bias':np.zeros(1)}
hyper_p={'alpha':alpha}

####################

LOG=logistic.LOGISTIC()
par,loss=LOG.sgd(X_train.copy(), y_train.copy(),start_p, hyper_p, eta=1e-5,epochs=1e4,batch_size=50,verbose=True)

y_pred=LOG.predict(X_test.copy(),par)
print(classification_report(y_test.copy(), y_pred))
print(confusion_matrix(y_test.copy(), y_pred))

print ('-------------------------------------------')
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(penalty="l2",solver="liblinear", C=1/alpha,fit_intercept=True)
log_reg.fit(X_train,y_train)
y_pred2 = log_reg.predict(X_test)
print(classification_report(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))