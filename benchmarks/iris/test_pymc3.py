import warnings
warnings.filterwarnings("ignore")

from sklearn import datasets
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor as tt
import sys 
sys.path.append("./") 


epochs = 1000
eta=1e-2
batch_size=100
alpha=1e-2

iris = datasets.load_iris()
data = iris.data  
labels = iris.target
classes=np.unique(iris.target)
X, y = iris.data, iris.target
X = (X - X.mean(axis=0)) / X.std(axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

with pm.Model() as model_s:
    alpha = pm.Normal('alpha', mu=0, sd=2, shape=3)
    beta = pm.Normal('beta', mu=0, sd=2, shape=(4,3))
    mu = alpha + pm.math.dot(X_train, beta)
    theta = tt.nnet.softmax(mu)

    yl = pm.Categorical('yl', p=theta, observed=y_train)
    hmc=pm.step_methods.hmc.hmc.HamiltonianMC(path_length=2.0,is_cov=True,adapt_step_size=False)
    trace_s = pm.sample(2000,step=hmc)


print(pm.summary(trace_s))
pm.traceplot(trace_s)
plt.show()