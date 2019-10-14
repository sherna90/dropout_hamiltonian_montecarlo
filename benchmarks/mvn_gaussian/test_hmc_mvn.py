import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set()
from importlib import reload
import sys

sys.path.append('./')

import hamiltonian.models.cpu.mvn_gaussian as model
import hamiltonian.inference.cpu.hmc as sampler
m=model.mvn_gaussian({'mu':np.zeros(2),'cov':np.array([[1.0, 0.8], [0.8, 1.0]])})
hmc=sampler.hmc(m,start_p={'x':np.random.rand(2)},path_length=1,step_size=0.1) 
samples,positions,momentums,logp=hmc.sample(100,100)

steps = slice(None, None, 20)
fig, ax = plt.subplots(figsize=(10,7))
ax.plot(0, 0, "o", color="w", ms=20, mfc="C1")

for pos,mom in zip(positions, momentums):
    q=np.array([q['x'] for q in pos])
    p=np.array([p['x'] for p in mom])
    ax.quiver(
        q[steps, 0],
        q[steps, 1],
        p[steps, 0],
        p[steps, 1],
        headwidth=4,
        scale=80,
        headlength=7,
        alpha=0.9,
    )
    ax.plot(q[:, 0], q[:, 1], "k-", lw=0.1)

ax.plot(samples['x'][:, 0], samples['x'][:, 1], "o", color="w", mfc="C2", ms=10)
ax.set_title("2D Gaussian trajectories!\nArrows show momentum!")
plt.show()

b_cols=columns=['x', 'y']
b_sample = pd.DataFrame(samples['x'],columns=b_cols)

g = sns.jointplot(x="x", y="y", data=b_sample, kind="kde", color="k", shade=True)
g.plot_joint(plt.scatter, c="r", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0.5)
g.set_axis_labels("$X$", "$Y$");
plt.show()