import numpy as np
import time
from scipy.spatial import distance
from scipy.cluster.hierarchy import single

from spcal.lib.spcalext import pdist_square, mst_linkage


np.random.seed(1231)
a = np.random.random(10000).reshape(1000, 10)

t0 = time.time()
r1 = distance.pdist(a)
t1 = time.time()

Z1 = single(r1);

print(r1, t1 - t0)

t0 = time.time()
r2 = pdist_square(a)
# r2 = np.sqrt(r2)

Z2, D = mst_linkage(r1, a.shape[0]);

# print(D.shape, Z1.shape)

# print(D[:50], D[-10:])
# print(Z1[:50, 3], Z1[-10:, 2])
# print(Z1[:10, 2], D[:10])
t1 = time.time()
print(r2, t1 - t0)

print(np.all(Z1[:, 3] == Z2[:, 2]))
