import time

import cupy as cp
import numpy as np

aux = time.time()
for i in range(1000):
    np.sum(np.arange(10))
print time.time() - aux

aux = time.time()
for i in range(1000):
    cp.sum(cp.arange(10))
print time.time() - aux
