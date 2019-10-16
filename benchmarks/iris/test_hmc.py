import warnings
warnings.filterwarnings("ignore")

from sklearn import datasets
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import sys 
sys.path.append('./')
import hamiltonian.utils as utils
import hamiltonian.models.cpu.softmax as base_model
import hamiltonian.inference.cpu.hmc as sampler

from importlib import reload

iris = datasets.load_iris()
data = iris.data  
labels = iris.target
classes=np.unique(iris.target)
X, y = iris.data, iris.target
X = (X - X.mean(axis=0)) / X.std(axis=0)
num_classes=len(classes)
y=utils.one_hot(y,num_classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,shuffle=True)

D=X_train.shape[1]
K=len(classes)


niter = 1e4
burnin=1e4
eta=0.1
alpha=1/100.

start_p={'weights':np.random.random((D,K)),
        'bias':np.random.random((K))}
hyper_p={'alpha':alpha}

model=base_model.softmax(hyper_p)
hmc=sampler.hmc(model,start_p,path_length=3,step_size=eta)
samples,loss,positions,momentums=hmc.sample(niter,burnin,None,X_train=X_train,y_train=y_train)

import pandas as pd
b_cols=[u+str(v) for u,v in zip(['b']*3,range(3))]
b_sample=pd.DataFrame(samples['bias'],columns=b_cols)
print(b_sample.describe())

post_par={var:np.median(samples[var],axis=0) for var in samples.keys()}
y_pred=model.predict(post_par,X_test)
print(classification_report(y_test.argmax(axis=1), y_pred))
print(confusion_matrix(y_test.argmax(axis=1), y_pred))


fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(10,7))
ax[0].plot(loss)
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Log-loss")
ax[1].hist(samples['bias'][:,0])
ax[1].hist(samples['bias'][:,1])
ax[1].hist(samples['bias'][:,2])
ax[1].set_xlabel("Bias")
ax[1].set_ylabel("Freq")
plt.show()

""" (test_pymc3.py:12470): Gtk-WARNING **: 17:42:50.400: Locale not supported by C library.
        Using the fallback 'C' locale.
WARNING (theano.tensor.blas): We did not find a dynamic library in the library_dir of the library we use for blas. If you use ATLAS, make sure to compile it with dynamics library.
Multiprocess sampling (4 chains in 4 jobs)
NUTS: [beta, alpha]
Sampling 4 chains: 100%|#######################################################################################| 6000/6000 [07:05<00:00, 14.09draws/s]
               mean        sd  mc_error   hpd_2.5  hpd_97.5        n_eff      Rhat
alpha__0  -0.014441  1.147441  0.020240 -2.347501  2.036146  2384.589183  1.000348
alpha__1  -0.098857  1.147234  0.020187 -2.397440  2.005050  2387.076299  1.000329
alpha__2   0.088583  1.146870  0.020172 -2.251247  2.147719  2386.451059  1.000324
beta__0_0  0.048657  1.149919  0.024850 -2.249878  2.203613  2153.765103  0.999761
beta__0_1  0.048392  1.150850  0.024813 -2.170810  2.285689  2159.872450  0.999779
beta__0_2  0.048276  1.150371  0.024753 -2.216239  2.224373  2164.297163  0.999729
beta__1_0 -0.005405  1.128446  0.019909 -2.115645  2.272844  2786.041049  1.000499
beta__1_1 -0.005355  1.128521  0.019894 -2.156494  2.234372  2783.871468  1.000474
beta__1_2 -0.005548  1.128457  0.019923 -2.109658  2.266280  2782.490030  1.000494
beta__2_0 -0.004907  1.192647  0.020895 -2.468583  2.238681  2576.223556  0.999911
beta__2_1 -0.004414  1.191844  0.021374 -2.242231  2.452480  2575.181436  0.999996
beta__2_2 -0.006487  1.193235  0.021208 -2.322853  2.409378  2587.128418  0.999824
beta__3_0  0.040402  1.172653  0.022603 -2.338267  2.272882  2988.843195  0.999560
beta__3_1  0.040275  1.172706  0.022470 -2.353356  2.269792  2998.609428  0.999556
beta__3_2  0.042598  1.169505  0.022553 -2.318622  2.292795  2966.418511  0.999546 

stan optim
Initial log joint probability = -154.447
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
      19      -5.28956         1.536      0.209354           1           1       25   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
      39      -5.17018     0.0196875      0.048719      0.6426      0.6426       46   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
      59      -5.16212    0.00998317    0.00733584           1           1       67   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
      79      -5.16191    0.00399693    0.00122098           1           1       88   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
      95       -5.1619   0.000350597   8.36027e-05      0.8737      0.8737      105  

>>> print(op['weights'])
[[-2.16871943  2.49226699 -0.32759529]
 [ 2.51667281 -0.4296626  -2.08424852]
 [-5.74229906 -4.28517259 10.02485957]
 [-5.15459595 -0.73729125  5.887566  ]]
>>> print(op['bias'])
[ 0.18269953  6.29433656 -6.4741582 ]


                b1           b2           b3
count  4000.000000  4000.000000  4000.000000
mean     -2.794040     9.738752    -7.282612
std       7.367385     6.303184     6.797948
min     -33.081503   -11.519215   -33.761193
25%      -7.527263     5.436548   -11.624500
50%      -2.646370     9.659394    -7.255833
75%       2.156013    14.024002    -2.812740
max      19.868073    35.792211    15.617151
"""