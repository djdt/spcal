import numpy as np

from spcal.cluster import (
    agglomerative_cluster,
    cluster_information,
    prepare_data_for_clustering,
    prepare_results_for_clustering,
)


def test_agglomerative_cluster():
    a = np.array(
        [[0.1, 0.2], [0.2, 0.3], [1.0, 1.4], [1.1, 1.5], [1.2, 1.5], [2.1, 2.2]]
    )

    T = agglomerative_cluster(a, 0.5)
    assert np.all(T == [1, 1, 2, 2, 2, 3])


def test_agglomerative_cluster_zero_and_one():
    a = np.array([[0.1]])
    T = agglomerative_cluster(a, 0.5)
    assert T == 0
    a = np.array([[]])
    T = agglomerative_cluster(a, 0.5)
    assert T.size == 0


def test_cluster_information():
    a = np.array(
        [[0.1, 0.2], [0.2, 0.3], [1.0, 1.4], [1.1, 1.5], [1.2, 1.5], [2.1, 2.2]]
    )
    means, stds, counts = cluster_information(a, np.array([1, 1, 2, 2, 2, 3]))

    assert np.allclose(means, [[1.1, 1.466667], [0.15, 0.25], [2.1, 2.2]])
    assert np.allclose(stds, [[0.08165, 0.04714], [0.05, 0.05], [0.0, 0.0]])
    assert np.all(counts == [3, 2, 1])

    means, stds, counts = cluster_information(a, np.array([0, 1, 2, 2, 2, 3]))
    assert np.all(counts == [2, 2, 1])


def test_prepare_data_for_clustering():
    a = np.array(
        [(1.0, 3.0), (2.0, 6.0), (3.0, 9.0)], dtype=[("a", float), ("b", float)]
    )

    pa = prepare_data_for_clustering(a)

    assert np.all(pa == (0.25, 0.75))

    d = np.stack((np.arange(5), np.ones(5)), axis=1)

    pd = prepare_data_for_clustering(d)

    assert np.allclose(pd[:, 0], np.arange(5) / (np.arange(5) + 1))
    assert np.allclose(pd[:, 1], np.ones(5) / (np.arange(5) + 1))


def test_preprare_results_for_clustering(random_result_generator):
    a = random_result_generator()
    b = random_result_generator()

    a.peak_indicies = np.array([0, 1, 2, 2, 5, 6, 8, 9, 10, 10])
    a.number_peak_indicies = 12
    b.peak_indicies = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    b.number_peak_indicies = 12

    x, valid = prepare_results_for_clustering([a, b], 12, "signal")
    assert x.shape == (12, 2)

    assert np.all(valid[:-1])
    assert not valid[-1]

    assert np.allclose(np.sum(x[valid], axis=1), 1.0)
