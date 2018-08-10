import warnings
warnings.filterwarnings("ignore")

import numpy as np

def one_hot(y,num_classes):
    encoding=np.zeros((len(y),num_classes))
    for i, val in enumerate(y):
        encoding[i, np.int(val)] = 1.0
    return encoding

def cross_entropy(y_hat, y):
    return np.sum(y * np.log(y_hat+1e-6))/y.shape[0]

def softmax(y_linear):
    exp = np.exp(y_linear-np.max(y_linear, axis=1).reshape((-1,1)))
    norms = np.sum(exp, axis=1).reshape((-1,1))
    return exp / norms

def net(X,par):
    ylinear = np.dot(X, par['weights']) + par['bias']
    yhat = softmax(ylinear)
    return yhat

def grad(X,y,par):
    yhat=net(X,par)
    diff = yhat-y
    grad_w = np.dot(X.T, diff)
    grad_b = np.sum(diff, axis=0)
    grad={}
    grad['weights']=grad_w/y.shape[0]+(0.5/y.shape[0])*par['alpha']*par['weights']
    dim=par['weights'].shape[0]
    grad['bias']=grad_b/y.shape[0]
    return grad	
    
def loss(X, y, par):
    y_hat=net(X,par)
    return -cross_entropy(y_hat,y)

def iterate_minibatches(X, y, batchsize):
    assert X.shape[0] == y.shape[0]
    for start_idx in range(0, X.shape[0] - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield X[excerpt], y[excerpt]

def sgd(X, y,num_classes, par,eta=1e-2,epochs=1e2,batch_size=20,scale=True,transform=True,verbose=True):
    loss_val=np.zeros((np.int(epochs)))
    dim=par['weights'].shape[0]
    momemtum={'weights':np.zeros((par['weights'].shape)),'bias':np.zeros((par['bias'].shape))}
    gamma=0.99
    for i in range(np.int(epochs)):
        for batch in iterate_minibatches(X, y, batch_size):
            X_batch, y_batch = batch
            if scale:
                X_batch=X_batch/255.
            if transform:
                y_batch=one_hot(y_batch,num_classes)
            grad_p=grad(X_batch,y_batch,par)
            momemtum['weights'] = gamma * momemtum['weights'] + eta * grad_p['weights']
            par['weights']-=momemtum['weights']
            momemtum['bias'] = gamma * momemtum['bias'] + eta * grad_p['bias']    
            par['bias']-=momemtum['bias']
            #print 'norm gradient: ',np.linalg.norm(par['weights'])
        loss_val[i]=loss(X_batch,y_batch,par)+(0.5/dim)*par['alpha']*np.sum(np.square(par['weights']))
        #eta *= (1. / (1. + decay * epochs))
        if verbose:
            print('loss: {0:.4f}'.format(loss(X_batch,y_batch,par)) )
    return par,loss_val

def predict(X,par,scale=True):
    if scale:
        X=X[:]/255.
    yhat=net(X,par)
    pred=yhat.argmax(axis=1)
    return pred	