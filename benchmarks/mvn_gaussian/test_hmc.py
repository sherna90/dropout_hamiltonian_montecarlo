import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from importlib import reload
import sys

sys.path.append('./')

import hamiltonian.models.cpu.gaussian as model
import hamiltonian.inference.cpu.hmc as sampler
m=model.gaussian({'mu':0,'sigma':0.1})
hmc=sampler.hmc(m,start_p={'x':np.random.rand()},path_length=1,step_size=0.1) 
samples,positions,momentums,logp=hmc.sample(100,500)

fig, ax = plt.subplots(figsize=(10,7))
for pos,mom in zip(positions,momentums):
    ax.plot([float(q['x']) for q in pos], [float(p['x']) for p in mom])

y_min, _ = ax.get_ylim()
ax.plot(samples['x'], y_min + np.zeros_like(samples['x']), "ko")
ax.set_xlabel("Position")
ax.set_ylabel("Momentum")
plt.show()