"""Agglomerative clustering."""

import numpy as np

from spcal.lib.spcalext.clustering import (
    cluster_by_distance,
    mst_linkage,
    pairwise_euclidean,
)


def prepare_data_for_clustering(data: np.ndarray | dict[str, np.ndarray]) -> np.ndarray:
    """Prepare data by stacking into 2D array.

    Takes a dictionary or structured array and creates an NxM array, where M is the
    number of names / keys and N the length of each array.

    Args:
        data: dictionary of names: array or structured array

    Returns:
        2D array, ready for ``agglomerative_cluster``
    """
    names = list(data.dtype.names if isinstance(data, np.ndarray) else data.keys())

    X = np.empty((len(data[names[0]]), len(names)), dtype=np.float64)
    for i, name in enumerate(names):
        X[:, i] = data[name]
    totals = np.sum(X, axis=1)
    np.divide(X.T, totals, where=totals > 0.0, out=X.T)
    return X


def agglomerative_cluster(X: np.ndarray, max_dist: float) -> np.ndarray:
    """Cluster data.

    Performs agglomerative clustering by merging close clusters until none are
    closer than ``max_dist``. Distance is measured as Euclidean distance.

    Args:
        X: 2D array (samples, features)
        max_dist: maximum distance between clusters

    Returns:
        cluster indicies
    """
    if X.size == 0:
        return np.array([], dtype=int)
    dists = pairwise_euclidean(X)
    Z, ZD = mst_linkage(dists, X.shape[0])
    T = cluster_by_distance(Z, ZD, max_dist) - 1
    return T


def cluster_information(
    X: np.ndarray, T: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get information about a clustering result.

    Clusters are sorted by size, largest to smallest.

    Args:
        X: 2D array (samples, features)
        T: cluster indicies

    Returns:
        cluster means
        cluster stds
        cluster counts
    """
    counts = np.bincount(T)
    means = np.empty((counts.size, X.shape[1]), dtype=np.float64)
    stds = np.empty((counts.size, X.shape[1]), dtype=np.float64)

    for i in range(means.shape[1]):
        sx = np.bincount(T, weights=X[:, i])
        sx2 = np.bincount(T, weights=X[:, i] ** 2)
        means[:, i] = np.divide(sx, counts, where=counts >= 1.0)
        var = np.divide(sx2, counts, where=counts >= 1.0) - means[:, i] ** 2
        stds[:, i] = np.sqrt(np.where(var > 0.0, var, 0.0))

    idx = np.argsort(counts)[::-1]
    return means[idx], stds[idx], counts[idx]
