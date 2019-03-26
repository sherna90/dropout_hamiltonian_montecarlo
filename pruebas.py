import numpy as np
import cupy as cp

posterior_sample = {'bias':np.array([[-0.08838913],
       [ 0.13665113],
       [-0.0371356 ]]),
       'weights': np.array([[-1.10161009, -0.48915963],
       [ 0.01455899, -0.14626398],
       [-0.91777321, -0.31616264]])}

 
par_mean={var:cp.mean(cp.asarray(posterior_sample[var]),axis=0) for var in posterior_sample.keys()}
par_var={var:cp.var(cp.asarray(posterior_sample[var]),axis=0) for var in posterior_sample.keys()}
print par_mean.keys()