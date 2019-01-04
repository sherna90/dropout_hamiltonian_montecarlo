import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sys
sys.path.append("./") 
import hamiltonian.logistic as logistic
import hamiltonian.utils as utils
import numpy as np



D=2
centers = [[-5, 0],  [5, -1]]
X, y = make_blobs(n_samples=1000, centers=centers, cluster_std=1,random_state=40)
X = (X - X.mean(axis=0)) / X.std(axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,shuffle=True)

alpha=1./4.
start_p={'weights':np.zeros(D),'bias':np.zeros(1)}
hyper_p={'alpha':alpha}
par,loss=logistic.sgd(X_train,y_train,start_p,hyper_p,eta=1e-5,epochs=1e4,batch_size=50,verbose=True)
y_pred=logistic.predict(X_test,par)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print par['weights']

print '-------------------------------------------'
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(penalty="l2",solver="liblinear", C=1/alpha,fit_intercept=True)
log_reg.fit(X_train,y_train)
y_pred2 = log_reg.predict(X_test)
print(classification_report(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))
print log_reg.coef_

plt.figure()
plt.scatter(X[:, 0], X[:, 1],marker='o', c=y,s=25, edgecolor='k')
plt.axis('tight')
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
coef = log_reg.coef_.flatten()
intercept = log_reg.intercept_.flatten()
liblinear_p={'weights':coef,'bias':intercept}
def plot_hyperplane(par, color):
    bd = lambda x0,par :  (-(x0 * par['weights'][0]) - par['bias']) / par['weights'][1]
    r=np.linspace(xmin,xmax)
    plt.plot(r,bd(r,par),ls="--", color=color)

plot_hyperplane(par, "r")
plot_hyperplane(liblinear_p, "c")
plt.show()