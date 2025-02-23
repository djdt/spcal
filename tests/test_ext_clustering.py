import numpy as np

from spcal.lib.spcalext.clustering import (
    cluster_by_distance,
    mst_linkage,
    pairwise_euclidean,
)


def test_pairwise_euclidean():
    a = np.arange(4.0).reshape(4, 1)
    assert np.all(pairwise_euclidean(a) == [1.00, 2.0, 3.0, 1.0, 2.0, 1.0])
    a = np.ones((9, 3))
    assert np.all(pairwise_euclidean(a) == 0.0)
    a = np.arange(16.0).reshape(4, 4)
    assert np.all(pairwise_euclidean(a) == [8.0, 16.0, 24.0, 8.0, 16.0, 8.0])


def test_mst_linkage():
    a = np.array([0.1, 1.0, 0.2, 1.2, 3.3, 2.3, 0.7, 0.75]).reshape(8, 1)
    dists = pairwise_euclidean(a)
    Z, ZD = mst_linkage(dists, 8)

    assert np.allclose(ZD, [0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 1.1])
    assert np.all(Z[0] == [6, 7])  # 0.05
    assert np.all(Z[1] == [0, 2])  # 0.1
    assert np.all(Z[2] == [1, 3])  # 0.2
    assert np.all(Z[3] == [8, 10])
    assert np.all(Z[4] == [9, 11])
    assert np.all(Z[5] == [4, 5])
    assert np.all(Z[6] == [12, 13])


def test_cluster_by_distance():
    a = np.array([0.1, 0.2, 0.3, 0.42, 0.5, 0.61, 0.72, 0.83]).reshape(8, 1)
    dists = pairwise_euclidean(a)
    Z, ZD = mst_linkage(dists, 8)
    T = cluster_by_distance(Z, ZD, 0.05)
    assert np.unique(T).size == 8  # all single clusters
    T = cluster_by_distance(Z, ZD, 0.1)
    assert np.all(T == [1, 1, 1, 2, 2, 3, 4, 5])
    T = cluster_by_distance(Z, ZD, 0.2)
    assert np.all(T == [1, 1, 1, 1, 1, 1, 1, 1])
