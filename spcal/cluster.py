"""Agglomerative clustering."""

from typing import TYPE_CHECKING
import numpy as np
import numpy.lib.recfunctions as rfn

from spcal.lib.spcalext.clustering import (
    cluster_by_distance,
    mst_linkage,
    pairwise_euclidean,
)

if TYPE_CHECKING:
    from spcal.processing.result import SPCalProcessingResult


def prepare_data_for_clustering(data: np.ndarray) -> np.ndarray:
    """Prepare data by stacking into 2D array.

    Takes a dictionary or structured array and creates an NxM array, where M is the
    number of names / keys and N the length of each array.

    Args:
        data: dictionary of names: array or structured array

    Returns:
        2D array, ready for ``agglomerative_cluster``
    """
    if data.dtype.names is not None:
        X = rfn.structured_to_unstructured(data, dtype=np.float32)
    else:
        X = data.astype(np.float32, copy=True)
    totals = np.sum(X, axis=1)
    np.divide(X.T, totals, where=totals > 0.0, out=X.T)
    return X


def prepare_results_for_clustering(
    results: list["SPCalProcessingResult"], number_peaks: int, key: str
) -> np.ndarray:
    """Prepare data by stacking into 2D array.

    Conveience method for list of results.

    Args:
        results: list of results, peak_indicies must be generated
        number_peaks: number of peaks

    Returns:
        2D array with length `number_peaks`, ready for ``agglomerative_cluster``

    See Also:
        ``prepare_data_from_clustering``
    """
    if any(result.peak_indicies is None for result in results):
        raise ValueError("cannot cluster, peak indidices have not been generated")

    peak_data = np.zeros((number_peaks, len(results)), np.float32)
    for i, result in enumerate(results):
        if result.peak_indicies is None:
            raise ValueError("cannot cluster, peak_indicies have not been generated")
        if not result.canCalibrate(key):
            continue
        np.add.at(
            peak_data[:, i],
            result.peak_indicies[result.filter_indicies],
            result.calibrated(key),
        )
    return prepare_data_for_clustering(peak_data)


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
    if X.size < 2:
        return np.zeros(X.size, dtype=int)
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
