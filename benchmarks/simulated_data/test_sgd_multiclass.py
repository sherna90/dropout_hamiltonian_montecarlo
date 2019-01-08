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
X, y = make_blobs(n_samples=1000, centers=centers, cluster_std=1,random_state=40)
y=utils.one_hot(y,K)
X = (X - X.mean(axis=0)) / X.std(axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,shuffle=True)
alpha=1./10.
start_p={'weights':np.zeros((D,K)),'bias':np.zeros((K))}
hyper_p={'alpha':alpha}
par,loss=softmax.sgd(X_train,y_train,len(centers),start_p,hyper_p,eta=1e-3,epochs=1e3)
y_pred=softmax.predict(X_test,par)
#print(classification_report(y_test.argmax(axis=1), y_pred))
#print(confusion_matrix(y_test.argmax(axis=1), y_pred))
print par['weights']
print '-------------------------------------------'
from sklearn.linear_model import LogisticRegression
softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=1/alpha,fit_intercept=True)
softmax_reg.fit(X_train,np.argmax(y_train,axis=1))
y_pred2 = softmax_reg.predict(X_test)
#print(classification_report(y_test.argmax(axis=1), y_pred2))
#print(confusion_matrix(y_test.argmax(axis=1), y_pred2))

sns.set()
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#Z = softmax_reg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = softmax.predict(np.c_[xx.ravel(), yy.ravel()],par)
Z = Z.reshape(xx.shape)
#plt.pcolormesh(xx, yy, Z, cmap=plt.cm.RdBu, antialiased=True,alpha=0.8)
#plt.imshow(Z, extent=(x_min, x_max, y_min, y_max), origin="lower")
cm = plt.cm.RdBu
ax = plt.subplot()
ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
plt.scatter(X[:, 0], X[:, 1], c=np.array(["r", "g", "b"])[np.argmax(y,axis=1)], edgecolors='k')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.tight_layout()
plt.show()