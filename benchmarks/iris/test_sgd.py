import warnings
warnings.filterwarnings("ignore")

from sklearn import datasets
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import sys 
sys.path.append("./") 
use_gpu=False
if use_gpu:
    import hamiltonian.softmax_gpu as softmax
else:
    import hamiltonian.softmax as softmax
import hamiltonian.utils as utils

epochs = 1e4
eta=1e-1
batch_size=50
alpha=1/4.

iris = datasets.load_iris()
data = iris.data  
labels = iris.target
classes=np.unique(iris.target)
X, y = iris.data, iris.target
X = (X - X.mean(axis=0)) / X.std(axis=0)
num_classes=len(classes)
y=utils.one_hot(y,num_classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,shuffle=True)


D=X_train.shape[1]
K=len(classes)-1
start_p={'weights':1e2*np.random.randn(D,K),
        'bias':1e2*np.random.randn(K)}
hyper_p={'alpha':alpha}
par,loss=softmax.sgd(X_train,y_train,num_classes,start_p,hyper_p,eta=eta,epochs=epochs,batch_size=batch_size,verbose=1)
y_pred=softmax.predict(X_test,par)
print(classification_report(y_test.argmax(axis=1), y_pred))
print(confusion_matrix(y_test.argmax(axis=1), y_pred))
print par['bias']
print '-------------------------------------------'
from sklearn.linear_model import LogisticRegression
softmax_reg = LogisticRegression(multi_class="multinomial", solver="newton-cg", C=1/alpha,fit_intercept=True)
softmax_reg.fit(X_train,np.argmax(y_train,axis=1))
y_pred2 = softmax_reg.predict(X_test)
print(classification_report(y_test.argmax(axis=1), y_pred2))
print(confusion_matrix(y_test.argmax(axis=1), y_pred2))
print softmax_reg.intercept_