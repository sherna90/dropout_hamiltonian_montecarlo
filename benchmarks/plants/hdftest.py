import h5py 
import numpy as np

f=h5py.File('sghmc_plants2_0.h5','r')

print(f.keys())

for aux in f['bias']:
    print(aux)

#mean = {var:np.sum(f[var],axis=0) for var in f.keys()}
#print(mean)