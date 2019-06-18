from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sys
sys.path.append("../../") 
import hamiltonian.cpu.logistic as logistic
import hamiltonian.cpu.hmc as sampler
import hamiltonian.utils as utils
import numpy as np

D=2
centers = [np.random.random_integers(0,10,D),np.random.random_integers(0,10,D)]
X, y = make_blobs(n_samples=100, centers=centers, cluster_std=10,random_state=40)
X = (X - X.mean(axis=0)) / X.std(axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

alpha=1./4.
start_p={'weights':np.zeros(D),'bias':np.zeros(1)}
hyper_p={'alpha':alpha}

LOG=logistic.LOGISTIC()
par,loss=LOG.sgd(X_train.copy(), y_train.copy(), start_p.copy(), hyper_p.copy(), eta=1e-5,epochs=1e4,batch_size=50,verbose=True)
y_pred=LOG.predict(X_test.copy(), par)
print(classification_report(y_test.copy(), y_pred))
print(confusion_matrix(y_test.copy(), y_pred))

mcmc=sampler.hmc(X_train.copy(),y_train.copy(),LOG.loss, LOG.grad, start_p.copy(),hyper_p.copy(), path_length=1,verbose=0)

#backend = 'test_sghmc_'
backend = None
niter = 1e3
burnin = 1e2

posterior_sample,logp_samples=mcmc.sample(niter,burnin, backend=backend)

if backend:
    par_mean = mcmc.backend_mean(posterior_sample, niter)

    y_pred_mc=LOG.predict(X_test.copy(),par_mean)

    print(classification_report(y_test.copy(), y_pred))
    print(confusion_matrix(y_test.copy(), y_pred))
else:
    post_par={var:np.mean(posterior_sample[var],axis=0).reshape(start_p[var].shape) for var in posterior_sample.keys()}
    post_par_var={var:np.var(posterior_sample[var],axis=0).reshape(start_p[var].shape) for var in posterior_sample.keys()}
    y_pred=LOG.predict(X_test.copy(),post_par)
    print(classification_report(y_test.copy(), y_pred))
    print(confusion_matrix(y_test.copy(), y_pred))