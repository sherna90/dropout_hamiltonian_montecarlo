from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import hamiltonian.utils as utils

import hamiltonian.softmaxcpu as softmax
import hamiltonian.pruebaSgldQueue as sampler

K = 5
D=100
centers = [np.random.random_integers(0,10,D) for i in range(K)]
X, y = make_blobs(n_samples=250, centers=centers, cluster_std=10,random_state=40)

y=utils.one_hot(y,K)

X = (X - X.mean(axis=0)) / X.std(axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

################################

alpha=1./10.
start_p={'weights':np.zeros((D,K)),'bias':np.zeros((K))}
hyper_p={'alpha':alpha}

SOFT=softmax.SOFTMAX()
par,loss=SOFT.sgd(X_train.copy(), y_train.copy(),K, start_p, hyper_p, eta=1e-5,epochs=1e4,batch_size=50,verbose=True)

y_pred=SOFT.predict(X_test.copy(),par)
print(classification_report(y_test.copy().argmax(axis=1), y_pred))
print(confusion_matrix(y_test.copy().argmax(axis=1), y_pred))
print ('-------------------------------------------')

mcmc=sampler.SGLD(X_train.copy(),y_train.copy(),SOFT.loss, SOFT.grad, start_p.copy(),hyper_p.copy(), path_length=1,verbose=0)

#backend = 'test_sghmc_'
backend = None
niter = 1e4
burnin = 1e3

posterior_sample,logp_samples=mcmc.multicore_sample(niter,burnin,batch_size=50, backend=backend)

post_par={var:np.mean(posterior_sample[var],axis=0).reshape(start_p[var].shape) for var in posterior_sample.keys()}
post_par_var={var:np.var(posterior_sample[var],axis=0).reshape(start_p[var].shape) for var in posterior_sample.keys()}
y_pred=SOFT.predict(X_test.copy(),post_par)
print(classification_report(y_test.copy().argmax(axis=1), y_pred))
print(confusion_matrix(y_test.copy().argmax(axis=1), y_pred))

print ('-------------------------------------------')
from sklearn.linear_model import LogisticRegression
softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=1/alpha,fit_intercept=True)
softmax_reg.fit(X_train.copy(),np.argmax(y_train.copy(),axis=1))
y_pred2 = softmax_reg.predict(X_test.copy())
print(classification_report(y_test.copy().argmax(axis=1), y_pred2))
print(confusion_matrix(y_test.copy().argmax(axis=1), y_pred2))