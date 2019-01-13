import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import sys
import time
import h5py
sys.path.append("./")

import hamiltonian.softmax as softmax
import hamiltonian.utils as utils

eta=1e-5
epochs=50
batch_size=50
alpha=1./100.
data_path = 'data/'

mnist_train=h5py.File('data/mnist_train.h5','r')
X_train=mnist_train['X_train'][:].reshape((-1,28*28))
X_train=X_train/255.
y_train=mnist_train['y_train']

mnist_test=h5py.File('data/mnist_test.h5','r')
X_test=mnist_test['X_test'][:].reshape((-1,28*28))
X_test=X_test/255.
y_test=mnist_test['y_test']


classes=np.unique(y_train)
D=X_train.shape[1]
K=len(classes)
y_train=utils.one_hot(y_train[:],K)
y_test=utils.one_hot(y_test[:],K)
import time

start_time=time.time()

start_p={'weights':np.zeros((D,K)),
        'bias':np.zeros((K))}
hyper_p={'alpha':alpha}
par,loss=softmax.sgd(X_train,y_train,K,start_p,hyper_p,eta=eta,epochs=epochs,batch_size=batch_size,verbose=1)
elapsed_time=time.time()-start_time 
print(elapsed_time)
y_pred=softmax.predict(X_test,par)
cnf_matrix=confusion_matrix(y_test[:].argmax(axis=1), y_pred)
print(classification_report(y_test[:].argmax(axis=1), y_pred))
print(cnf_matrix)

import matplotlib.pyplot as plt 
import seaborn as sns
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=np.int32(classes),title='Confusion matrix')
plt.savefig('mnist_confusion_matrix.pdf',bbox_inches='tight')
plt.close()

sns.set()
plt.figure()
plt.plot(range(epochs),loss)
#plt.title('Training loss')
plt.ylabel('log-loss')
plt.xlabel('epochs')
plt.savefig('minist_fine_tuning.pdf',bbox_inches='tight',dpi=100)
plt.close()

mnist_test.close()
mnist_train.close()

