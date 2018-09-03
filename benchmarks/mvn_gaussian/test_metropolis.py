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

import hamiltonian.metropolis as sampler

start_p={'mu':10*np.random.randn(2)}
hyper_p={'cov':np.array([[1.0,0.8],[0.8,1.0]])}
mcmc=sampler.MH(np.array(2),np.array(0),mvn_gaussian.loss, start_p,hyper_p,scale=False,transform=False,verbose=1)
posterior_sample=mcmc.sample(1e3,1e2)
post_par={'mu':np.mean(posterior_sample['mu'],axis=0).reshape(start_p['mu'].shape)}

b_cols=columns=['x', 'y']
b_sample = pd.DataFrame(posterior_sample['mu'], columns=b_cols)
print start_p
print b_sample.mean()

print b_sample
sns.set(style="ticks")
g = sns.jointplot(x="x", y="y", data=b_sample, kind="kde", color="k")
g.plot_joint(plt.scatter, c="r", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$X$", "$Y$");
plt.show()