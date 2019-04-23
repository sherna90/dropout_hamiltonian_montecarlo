import cupy as cp
import numpy as np
import time as t

a = np.arange(10000)
b = 2
a2 = cp.asarray(a)
b2 = cp.asarray(2)

aux = t.time()
np.dot(a,b)
np.sum(a)
print("CPU: ", t.time()-aux)


aux = t.time()
cp.dot(a2,b2)
cp.sum(a2)
print("GPU con asarray: ",t.time()-aux)


aux = t.time()
cp.dot(a,b)
cp.sum(a)
print("GPU con cpu", t.time()-aux)
