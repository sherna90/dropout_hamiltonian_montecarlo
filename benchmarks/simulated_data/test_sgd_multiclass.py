import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sys
sys.path.append("./") 
import hamiltonian.softmax as softmax
import hamiltonian.utils as utils
import numpy as np



D=2
centers = [[-5, 0],  [5, -1], [10,10]]
K=len(centers)
print K
X, y = make_blobs(n_samples=1000, centers=centers, cluster_std=1,random_state=40)
y=utils.one_hot(y,K)
X = (X - X.mean(axis=0)) / X.std(axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,shuffle=True)
alpha=1./4.
start_p={'weights':np.zeros((D,K)),'bias':np.zeros((K))}
hyper_p={'alpha':alpha}
par,loss=softmax.sgd(X_train,y_train,len(centers),start_p,hyper_p,eta=1e-2,epochs=1e3)
y_pred=softmax.predict(X_test,par)
print(classification_report(y_test.argmax(axis=1), y_pred))
print(confusion_matrix(y_test.argmax(axis=1), y_pred))
print par['weights']
print '-------------------------------------------'
from sklearn.linear_model import LogisticRegression
softmax_reg = LogisticRegression(multi_class="multinomial", solver="newton-cg", C=1/alpha,fit_intercept=True)
softmax_reg.fit(X_train,np.argmax(y_train,axis=1))
y_pred2 = softmax_reg.predict(X_test)
print(classification_report(y_test.argmax(axis=1), y_pred2))
print(confusion_matrix(y_test.argmax(axis=1), y_pred2))

sns.set()
plt.figure()
plt.scatter(X[:, 0], X[:, 1],marker='o', c=y,s=25, edgecolor='k')
plt.axis('tight')
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
coef = softmax_reg.coef_.flatten()
print softmax_reg.coef_
intercept = softmax_reg.intercept_.flatten()
liblinear_p={'weights':coef,'bias':intercept}
def plot_hyperplane(par, color):
    bd = lambda x0,par,i :  (-(x0 * par['weights'][0,i]) - par['bias'][i]) / par['weights'][1,i]
    r=np.linspace(xmin,xmax)
    for j in range(K):
        plt.plot(r,bd(r,par,j),ls="--", color=color)

plot_hyperplane(par, "r")
#plot_hyperplane(liblinear_p, "c")
plt.show()