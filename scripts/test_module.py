import numpy as np
import time
from scipy.spatial import distance

from spcal.lib.spcalext import pdist


a = np.random.random(10000).reshape(1000, 10)

t0 = time.time()
r1 = distance.pdist(a)
t1 = time.time()

print(r1, t1 - t0)

t0 = time.time()
r2 = pdist(a)
t1 = time.time()
r2 = np.sqrt(r2)
print(r2, t1 - t0)
