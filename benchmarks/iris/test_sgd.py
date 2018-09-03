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

epochs = 1000
eta=1e-2
batch_size=30
alpha=1e-3

iris = datasets.load_iris()
data = iris.data  
labels = iris.target
classes=np.unique(iris.target)
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


D=X_train.shape[1]
num_classes=len(classes)
start_p={'weights':np.random.randn(D,num_classes),
        'bias':np.random.randn(num_classes)}
hyper_p={'alpha':alpha}
par,loss=softmax.sgd(X_train,y_train,num_classes,start_p,hyper_p,eta=eta,epochs=epochs,batch_size=batch_size,scale=False,transform=True,verbose=1)
y_pred=softmax.predict(X_test,par,False)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print par