from sklearn import datasets
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def naive_hpd(post):
    sns.kdeplot(post)
    HPD = np.percentile(post, [2.5, 97.5])
    plt.plot(HPD, [0, 0], label='HPD {:.2f} {:.2f}'.format(*HPD), 
      linewidth=8, color='k')
    plt.legend(fontsize=16);
    plt.xlabel(r'$\theta$', fontsize=14)
    plt.gca().axes.get_yaxis().set_ticks([])

def logSigmoid(x,deriv=False):
	if(deriv==True):
		return self.sigmoid(x)*(1-self.sigmoid(x))
	#return 1/(1+np.exp(-x))
	return -np.log(1+np.exp(-x))

def sigmoid(x):
	phi = np.zeros(x.shape[0])
	for i in xrange(0,x.shape[0]):
		if (x[i] >0):
			phi[i] = 1.0/(1.0+np.exp(-x[i]))
		else:
			phi[i] = np.exp(x[i])/(1.0+np.exp(x[i]))
	return phi

def logLikelihood(x,X_train,Y_train):
	eta = X_train.dot(x)
	phi = sigmoid(eta)
	ll = 0.0
	for i in xrange(0,X_train.shape[0]):
		sg = phi[i]
		if(Y_train[i]>0):
			ll = ll+np.log(sg)
		else:
			ll = ll+np.log(1-sg)
	return ll

def prior(x, delta):
	return -(delta/x.shape[0]) * (np.linalg.norm(x, ord=2) ** 2)

def computeGradient(x, X_train, Y_train, delta):
	eta = X_train.dot(x)
	phi = sigmoid(eta)
	E_d = np.zeros(X_train.shape[1])
	for i in xrange(0,X_train.shape[0]):
		sg = phi[i]
		if(Y_train[i]>0):
			E_d = E_d + (1.0-sg)*(X_train[i,:])
		else:
			E_d = E_d + (0.0-sg)*(X_train[i,:])
	E_w = -(-2.0*delta/X_train.shape[1])*(x)
	return E_d+E_w

def logPosterior(x, X_train, Y_train, delta):
	loglikelihood = logLikelihood(x,X_train,Y_train)
	logPrior = prior(x, delta)
	return loglikelihood +logPrior

def predict(x,X_test):
	eta = X_test.dot(x)
	phi = sigmoid(eta)
	pred = np.zeros(X_test.shape[0])
	pred[phi > 0.5] = 1 
	pred[phi <= 0.5] = -1
	return pred

# run the sampler
import hmc2 as sampler
# import some data to play with
iris = datasets.load_iris()
data = iris.data  # we only take the first two features.
labels = iris.target

labels[labels == 1] = -1
labels[labels == 2] = -1
labels[labels == 0] = 1

index = np.random.permutation(data.shape[0])

data_new = data[index]
labels_new = labels[index]
#data_new = data
#labels_new = labels

X_train = data_new[0:100]
Y_train = labels_new[0:100]
X_test = data_new[100:150]
Y_test = labels_new[100:150]

delta = 1.0
x0 = np.random.randn(X_train.shape[1])

samples = sampler.hmc(logPosterior, x0, computeGradient, args=(X_train,Y_train,delta,),
                  nsamples=10**4,nomit=10**3,steps=10,stepadj=0.01)

mean_samples = samples.mean(axis = 0)

pred = predict(mean_samples, X_test)

print(classification_report(Y_test, pred))
print(confusion_matrix(Y_test, pred))
