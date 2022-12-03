from typing import Dict, Tuple

import numpy as np

from spcal.lib.spcalext import cluster_by_distance, mst_linkage, pairwise_euclidean


def prepare_data_for_clustering(data: np.ndarray | Dict[str, np.ndarray]) -> np.ndarray:
    names = list(data.dtype.names if isinstance(data, np.ndarray) else data.keys())

    X = np.empty((len(data[names[0]]), len(names)), dtype=np.float64)
    for i, name in enumerate(names):
        X[:, i] = data[name]
    totals = np.sum(X, axis=1)
    np.divide(X.T, totals, where=totals > 0.0, out=X.T)
    return X


def agglomerative_cluster(
    X: np.ndarray, max_dist: float
) -> Tuple[np.ndarray, np.ndarray]:
    dists = pairwise_euclidean(X)
    Z, ZD = mst_linkage(dists, X.shape[0])
    T = cluster_by_distance(Z, ZD, max_dist) - 1  # start ids at 0

    counts = np.bincount(T)
    means = np.empty((counts.size, X.shape[1]), dtype=np.float64)

    for i in range(means.shape[1]):
        means[:, i] = np.bincount(T, weights=X[:, i]) / counts

    idx = np.argsort(counts)[::-1]
    return means[idx], counts[idx]
