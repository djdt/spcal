import numpy as np
import time
from scipy.spatial import distance
from scipy.cluster.hierarchy import single, fcluster

from spcal.lib.spcalext import pairwise_euclidean, mst_linkage, cluster_by_distance


np.random.seed(1231)
a = np.random.random(10000).reshape(1000, 10)

t0 = time.time()
r1 = distance.pdist(a)

Z1 = single(r1)
T1 = fcluster(Z1, 0.7, criterion="distance")
t1 = time.time()


print(t1 - t0)
# print(maxdists(Z1)[495:505], flush=True)
# print(maxdists(Z1)[-10:], flush=True)

t0 = time.time()
r2 = pairwise_euclidean(a)

Z2, D = mst_linkage(r1, a.shape[0])

T2 = cluster_by_distance(Z2, D, 0.7)
t1 = time.time()

# print(D.shape, Z1.shape)

# print(D[:50], D[-10:])
# print(Z1[:50, 3], Z1[-10:, 2])
# print(Z1[:10, 2], D[:10])
print(t1 - t0)

print(np.all(r1 == r2))
print(np.all(Z1[:, 0] == Z2[:, 0]))
print(np.all(Z1[:, 1] == Z2[:, 1]))
print(np.all(Z1[:, 3] == Z2[:, 2]))
print(np.all(Z1[:, 2] == D))
print(np.all(T1 == T2))
# print(T1[:20], T2[:20])
# print(T2)
