import warnings
warnings.filterwarnings("ignore")

from sklearn import datasets
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import softmax as softmax

lb = preprocessing.LabelBinarizer()
iris = datasets.load_iris()
data = iris.data  
labels = iris.target
classes=np.unique(iris.target)

data_new = data
labels_new = labels
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

scaler = preprocessing.StandardScaler().fit(X_train)
scaler.transform(X_train)
scaler.transform(X_test)
y_train = lb.fit_transform(y_train)
y_test = y_test


D=X_train.shape[1]
K=len(classes)
start_p={'weights':np.random.randn(D,K),'bias':np.random.randn(K),'alpha':1e-3}
par,loss=softmax.sgd(X_train,y_train,start_p,eta=1e-1,iter=1e5)
y_pred=softmax.predict(X_test,par)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
