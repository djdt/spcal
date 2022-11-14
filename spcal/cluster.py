import numpy as np

from spcal.lib.spcalext import pairwise_euclidean, mst_linkage, cluster_by_distance


def agglomerative_cluster(X: np.ndarray, max_dist: float):
    dists = pairwise_euclidean(X)
    Z, ZD = mst_linkage(dists, X.shape[0])
    T = cluster_by_distance(Z, ZD, max_dist) - 1  # start ids at 0

    counts = np.bincount(T)
    means = np.empty((counts.size, X.shape[1]), dtype=np.float64)

    for i in range(means.shape[1]):
        means[:, i] = np.bincount(T, weights=X[:, i]) / counts

    idx = np.argsort(counts)[::-1]
    return means[idx], counts[idx]
