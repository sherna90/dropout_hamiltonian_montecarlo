import numpy as numpy

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import seaborn as sns
import sys 
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append("./") 

use_gpu=False
import hamiltonian.mvn_gaussian as mvn_gaussian

import hamiltonian.hmc as hmc

start_p={'mu':10*np.random.randn(2)}
hyper_p={'cov':np.array([[1.0,0.8],[0.8,1.0]])}
mcmc=hmc.HMC(np.array(2),np.array(0),mvn_gaussian.loss, mvn_gaussian.grad, start_p,hyper_p, n_steps=10,scale=False,transform=False,verbose=1)
posterior_sample=mcmc.sample(1e3,1e2)
post_par=start_p={'mu':np.mean(posterior_sample['mu'],axis=0).reshape(start_p['mu'].shape)}

b_cols=columns=['m1', 'm2']
b_sample = pd.DataFrame(posterior_sample['mu'], columns=b_cols)
print "mean bias : ",b_sample.mean()
print "var bias : ",b_sample.var()
for col in b_cols:
    sns.kdeplot(b_sample[col], shade=True)
plt.show()