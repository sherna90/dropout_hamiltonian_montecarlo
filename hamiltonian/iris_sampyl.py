import pandas as pd
import sampyl as smp 
import seaborn as sns
from sampyl import np
import matplotlib.pyplot as plt

'''
iris = sns.load_dataset("iris")
df = iris.query("species == ('setosa', 'versicolor')")
x_n = 'sepal_length' 
X_train = np.reshape(df[x_n].values,(df.shape[0],1))
n=X_train.shape[0]
x_0=np.hstack([np.ones((n,1)),X_train])
y_0 = pd.Categorical(df['species']).codes

'''
iris = sns.load_dataset("iris")
df = iris.query("species == ('setosa', 'versicolor')")
y_0 = pd.Categorical(df['species']).codes
x_n = ['sepal_length', 'sepal_width']
x_0 = df[x_n].values
x_0 = (x_0 - x_0.mean(axis=0)) / x_0.std(axis=0)
#n= x_0.shape[0]
#x_0 = np.hstack([np.ones((n,1)),x_0])


weights = np.zeros(x_0.shape[1])
b = 1.0
eps = 1e-15
alpha=0.1
delta=0.05
niter=10000

def wTx(w, x):
    return np.dot(x, w)

def sigmoid(z):
    return 1./(1+np.exp(-z))

def logistic_predictions(w, x, b):
    predictions = sigmoid(wTx(w, x)+b)
    return predictions.clip(eps, 1.0-eps)

def custom_loss(y, y_predicted,w):
    #log_prior=-np.log(np.sqrt(2*np.pi*alpha))-np.dot(w.T,w)/(2*alpha)
    #return (y*np.log(y_predicted) + (1-y)*np.log(1-y_predicted)).mean()+log_prior
    #return -(y*np.log(y_predicted) + (1-y)*np.log(1-y_predicted)).mean()+0.5*alpha*np.dot(w.T,w)
    return (y*np.log(y_predicted) + (1-y)*np.log(1-y_predicted)).mean() - (np.dot(w.T,w)/(2*(alpha**len(w))))

def logp(w,b):
    y_predicted = logistic_predictions(w, x_0, b)
    return -custom_loss(y_0, y_predicted,w)

hmc = smp.Hamiltonian(logp, start={'w': weights, 'b': b},step_size=delta,n_steps=10)
chain = hmc.sample(niter, burn=niter/10,progress_bar=True)

#plt.scatter(chain.w[:,0],chain.w[:,1])
#plt.show()

df_cplus = pd.DataFrame([(chain.w[niter/10:,0]).tolist(), (chain.w[niter/10:,1]).tolist()]).T
sns.set(style="ticks", color_codes=True)
g2 = sns.pairplot(df_cplus)
sns.plt.show()