import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import sys
sys.path.append('./')
import time
import h5py
from tqdm import tqdm, trange

use_gpu=False
if use_gpu:
    import hamiltonian.softmax_gpu as softmax
else:
    import hamiltonian.softmax as softmax

plants_features=h5py.File('data/grape_esca_domain.h5','r')

npz=np.load("results/sgd_plants.npz")
par_sgd={}
par_sgd.update({0.1:npz['par_sgd_dropout_05'][()]})
par_sgd.update({0.5:npz['par_sgd_dropout_01'][()]})
par_sgd.update({0.9:npz['par_sgd_dropout_09'][()]})

n_samples=1000
class_prob={k:[] for k in par_sgd.keys()}
predictive_accuracy={k:[] for k in par_sgd.keys()}
ellapsed_time={}

for p,par in par_sgd.iteritems():
    t0=time.clock()
    test=plants_features.keys()[0]
    X_test=plants_features[str(test)]
    print('MC Dropout test')
    for k in tqdm(range(n_samples)):
        y_pred=softmax.predict_stochastic(X_test,par,prob=True,p=p)
        predictive_accuracy[p].append(np.argmax(y_pred))
        class_prob[p].append(y_pred)
    t1=time.clock()
    eps=t1-t0
    ellapsed_time[p]=eps
    print("MC dropout Ellapsed Time : ",eps)

posterior_sample=h5py.File('results/sgmcmc_plants_1000.h5','r')
mcmc_samples=posterior_sample['weights'].shape[0]
hyper_p={'alpha':1e-2}
predictive_accuracy.update({'sgld':[]})
class_prob.update({'sgld':[]})
t0=time.clock()
print('MCMC test')
for i in tqdm(range(1,mcmc_samples)):
    par={'weights':posterior_sample['weights'][i,:],'bias':posterior_sample['bias'][i,:]}
    test=plants_features.keys()[0]
    X_test=plants_features[str(test)]
    y_pred=softmax.predict(X_test,par,prob=True)
    predictive_accuracy['sgld'].append(np.argmax(y_pred))
    class_prob['sgld'].append(y_pred)
t1=time.clock()
eps=t1-t0
ellapsed_time['sgld']=eps
print("SGLD Ellapsed Time : ",eps)

np.savez("results/results_domian_plants.npz",predictive_accuracy=predictive_accuracy,class_prob=class_prob)

#import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set_style("whitegrid")

#for m in class_prob.keys():
#    fig = plt.figure()
#    df_prob=pd.DataFrame(np.vstack(np.log(class_prob[m])))
#    sns.boxplot(data=df_prob,orient="h",fliersize=0.5)
#    plt.savefig('domain_adaption_'+str(m).replace('.','')+'.pdf',bbox_inches='tight',dpi=200)
    
posterior_sample.close()
plants_features.close()
