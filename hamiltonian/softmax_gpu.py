import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pycuda.driver as cuda
from pycuda import gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.cumath as cumath
import pycuda.curandom as curandom
import time as t

import scikits.cuda.linalg as linalg
import scikits.cuda.misc as misc

linalg.init()
misc.init()

########## BEGIN *GPU* CODE ##########
# Create a softmax kernel to be used for the GPU-version of softmax(XB)
gpu_kernel = SourceModule("""
// M and N are dimensions of input matrix (rows by columns)
__global__ void softmax(float *output, int M, int N)
{
	 #include <math.h>
      int row = blockIdx.y*blockDim.y + threadIdx.y;
      float sum = 0;      
      if(row < M) {// && col < N) {
          // This is done to ensure numerical stability
          float max = output[row*N];
          for(int i=0;i<N;i++){
             float val = output[row*N + i];
             if(val > max) {max = val;}
          }          
          for(int i=0;i<N;i++){
             sum += exp(output[row*N + i]-max);
          }
   	    for(int i=0;i<N;i++){
             output[row*N + i] = exp(output[row*N + i]-max)/sum;
	    }	           
        }                 
                  
}
""")

def one_hot(y,num_classes):
    encoding=np.zeros((len(y),num_classes))
    for i, val in enumerate(y):
        encoding[i, np.int(val)] = 1.0
    return encoding

def cross_entropy(y_hat, y):
    prod = misc.multiply(y,cumath.log(y_hat+1e-6))
    return -misc.sum(prod)/y.shape[0]

def softmax(y_linear_gpu):
    softmax_kernel = gpu_kernel.get_function("softmax")
    grid2 = (y_linear_gpu.shape[0]+32-1)/32
    M = np.int32(y_linear_gpu.shape[0])       
    N = np.int32(y_linear_gpu.shape[1])
    #Perform softmax using GPU      
    softmax_kernel(y_linear_gpu, M, N, block=(1,32,1),grid=(1,grid2))

def net(X_gpu,par_gpu):
    Xw = linalg.dot(X_gpu,par_gpu['weights'])
    yhat = misc.add_matvec(Xw,par_gpu['bias'])
    softmax(yhat)
    return yhat

def grad(X_gpu,y_gpu,par_gpu):
    yhat=net(X_gpu,par_gpu)
    diff = yhat-y_gpu
    Xw = linalg.dot(X_gpu, diff,transa='T')
    alpha_gpu=par_gpu['alpha']*misc.ones_like(par_gpu['weights'])
    grad_l2=misc.multiply(alpha_gpu,par_gpu['weights'])
    dim=par_gpu['weights'].shape[0]
    grad_w = misc.add_matvec(Xw/y_gpu.shape[0],(0.5/y_gpu.shape[0])*grad_l2)
    grad_b = misc.sum(diff, axis=0)
    grad={}
    grad['weights']=grad_w
    grad['bias']=grad_b/y_gpu.shape[0]
    return grad	
    
def loss(X_gpu, y_gpu, par_gpu):
    y_hat=net(X_gpu,par_gpu)
    log_like=cross_entropy(y_hat,y_gpu)
    dim=par_gpu['weights'].shape[0]
    return log_like.get()+(0.5/dim)*par_gpu['alpha']*np.sum(np.square(par_gpu['weights'].get()))

def iterate_minibatches(X, y, batchsize):
    assert X.shape[0] == y.shape[0]
    for start_idx in range(0, X.shape[0] - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield X[excerpt], y[excerpt]

def sgd(X, y, num_classes,par,eta=1e-2,epochs=1e2,batch_size=20,scale=True,verbose=True):
    loss_val=np.zeros((np.int(epochs)))
    par_gpu={'weights':gpuarray.to_gpu(par['weights'].astype(np.float32).copy()),
        'bias':gpuarray.to_gpu(par['bias'].astype(np.float32).copy()),
        'alpha':np.float32(par['alpha']).copy()}
    momemtum={'weights':misc.zeros((par['weights'].shape),dtype=np.float32),
        'bias':misc.zeros((par['bias'].shape),dtype=np.float32)}
    gamma=0.9
    for i in range(np.int(epochs)):
        for batch in iterate_minibatches(X, y, batch_size):
            X_batch, y_batch = batch
            if scale:
                X_batch=X_batch/255.
                y_batch=one_hot(y_batch,num_classes)
            X_batch_gpu = gpuarray.to_gpu(X_batch.astype(np.float32).copy())
            y_batch_gpu = gpuarray.to_gpu(y_batch.astype(np.float32).copy())
            grad_p=grad(X_batch_gpu,y_batch_gpu,par_gpu)
            gamma_gpu=gamma*misc.ones_like(momemtum['weights'])
            eta_gpu=eta*misc.ones_like(grad_p['weights'])
            momemtum['weights']=misc.multiply(gamma_gpu,momemtum['weights'])+misc.multiply(eta_gpu,(grad_p['weights']))
            par_gpu['weights']-=momemtum['weights']
            gamma_bias_gpu=gamma*misc.ones_like(momemtum['bias'])
            eta_bias_gpu=eta*misc.ones_like(grad_p['bias'] )
            momemtum['bias'] = misc.multiply(gamma_bias_gpu,momemtum['bias'])+misc.multiply(eta_bias_gpu,(grad_p['bias']))  
            par_gpu['bias']-=momemtum['bias']
        #eta *= (1. / (1. + decay * epochs))
        if verbose:
            print('loss: {0:.4f}'.format(loss(X_batch_gpu,y_batch_gpu,par_gpu)) )
            #print('par',par_gpu['bias'],par_gpu['weights'])
        loss_val[i]=loss(X_batch_gpu,y_batch_gpu,par_gpu)
    par['weights']=par_gpu['weights'].get()
    par['bias']=par_gpu['bias'].get()
    return par,loss_val

def predict(X,par,scale=True):
    par_gpu={'weights':gpuarray.to_gpu(par['weights'].astype(np.float32).copy()),
        'bias':gpuarray.to_gpu(par['bias'].astype(np.float32).copy()),
        'alpha':np.float32(par['alpha']).copy()}
    if scale:
        X=X[:]/255.
    X_gpu = gpuarray.to_gpu(X[:].astype(np.float32).copy())
    yhat=net(X_gpu,par_gpu)
    pred=misc.argmax(yhat,axis=1)
    return pred.get()	