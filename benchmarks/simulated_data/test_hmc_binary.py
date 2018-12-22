import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sys
sys.path.append("./") 
import hamiltonian.logistic as logistic
import hamiltonian.hmc as hmc
import hamiltonian.utils as utils
import numpy as np



D=2
centers = [[-5, 0],  [5, -1]]
X, y = make_blobs(n_samples=1000, centers=centers, cluster_std=1,random_state=40)
X = (X - X.mean(axis=0)) / X.std(axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,shuffle=True)

alpha=1./4.
start_p={'weights':np.random.randn(D),'bias':np.random.randn(1)}
hyper_p={'alpha':alpha}
mcmc=hmc.HMC(X_train,y_train,logistic.loss, logistic.grad, start_p,hyper_p, path_length=1,step_size=0.08,verbose=1)
posterior_sample=mcmc.sample(1e3,1e2)
post_par={var:np.mean(posterior_sample[var],axis=0).reshape(start_p[var].shape) for var in posterior_sample.keys()}
post_par_var={var:np.var(posterior_sample[var],axis=0).reshape(start_p[var].shape) for var in posterior_sample.keys()}
y_pred=logistic.predict(X_test,post_par)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print post_par['weights']

sns.set()
plt.figure()
plt.scatter(X[:, 0], X[:, 1],marker='o', c=y,s=25, edgecolor='k')
plt.axis('tight')
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
def plot_hyperplane(par,var, color):
    bd = lambda x0,par :  (-(x0 * par['weights'][0]) - par['bias']) / par['weights'][1]
    bd_hpd_m = lambda x0,par,var :  (-(x0 * (par['weights'][0]-var['weights'][0]) - (par['bias']-var['bias']))) / (par['weights'][1]-var['weights'][1])
    bd_hpd_p = lambda x0,par,var :  (-(x0 * (par['weights'][0]+var['weights'][0]) - (par['bias']+ var['bias']))) / (par['weights'][1]+var['weights'][1])
    r=np.linspace(xmin,xmax)
    plt.plot(r,bd(r,par),ls="--", color=color)
    plt.fill_between(r, bd_hpd_m(r,par,var), bd_hpd_p(r,par,var), color=color, alpha=0.5)
    
plot_hyperplane(post_par, post_par_var,"b")
plt.show()