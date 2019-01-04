import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import sys
sys.path.append("./")
import hamiltonian.logistic as logistic
import hamiltonian.hmc as hmc
import hamiltonian.utils as utils
import numpy as np


X = pd.read_csv('data/german_data.csv', header=None)
y = pd.read_csv('data/german_labels.csv', header=None)
X = X.values
y = y.values
X = (X - X.mean(axis=0)) / X.std(axis=0)
y = y.reshape(-1)
D = X.shape[1]
alpha = 1.
start_p = {'weights': np.zeros(D), 'bias': np.zeros(1)}
hyper_p = {'alpha': alpha}
mcmc = hmc.HMC(X, y, logistic.log_likelihood, logistic.grad,start_p, hyper_p, path_length=1, verbose=1)
posterior_sample,logp_samples = mcmc.sample(1e3, 1e2)
post_par={var:np.mean(posterior_sample[var],axis=0).reshape(start_p[var].shape) for var in posterior_sample.keys()}
#post_par_var={var:np.var(posterior_sample[var],axis=0).reshape(start_p[var].shape) for var in posterior_sample.keys()}

plt.hist(logp_samples)
plt.show()
