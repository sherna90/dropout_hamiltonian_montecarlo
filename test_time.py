import time
import numpy as np
import cupy as cp

aux = time.time()
np.dot(3, 4)
print time.time() - aux