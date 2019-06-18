from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import sys
sys.path.append("../../")
import hamiltonian.utils as utils
import hamiltonian.cpu.softmax as softmax

num_classes = 3
D=2
centers = [np.random.random_integers(0,10,D) for i in range(num_classes)]

X, y = make_blobs(n_samples=1000, centers=centers, cluster_std=1,random_state=40)
y=utils.one_hot(y,num_classes)
X = (X - X.mean(axis=0)) / X.std(axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
alpha=1./4.
start_p={'weights':np.zeros((D,num_classes)),'bias':np.zeros((num_classes))}
hyper_p={'alpha':alpha}

################################

SOFT=softmax.SOFTMAX()
par,loss=SOFT.sgd(X_train.copy(), y_train.copy(),num_classes, start_p, hyper_p, eta=1e-5,epochs=1e2,batch_size=50,verbose=True)

y_pred=SOFT.predict(X_test.copy(),par)
print(classification_report(y_test.copy().argmax(axis=1), y_pred))
print(confusion_matrix(y_test.copy().argmax(axis=1), y_pred))

print ('-------------------------------------------')
from sklearn.linear_model import LogisticRegression
softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=1/alpha,fit_intercept=True)
softmax_reg.fit(X_train.copy(),np.argmax(y_train.copy(),axis=1))
y_pred2 = softmax_reg.predict(X_test.copy())
print(classification_report(y_test.copy().argmax(axis=1), y_pred2))
print(confusion_matrix(y_test.copy().argmax(axis=1), y_pred2))