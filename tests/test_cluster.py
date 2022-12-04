import numpy as np

from spcal.cluster import agglomerative_cluster, prepare_data_for_clustering


def test_agglomerative_cluster():
    a = np.array(
        [[0.1, 0.2], [0.2, 0.3], [1.0, 1.4], [1.1, 1.5], [1.2, 1.5], [2.1, 2.2]]
    )

    means, stds, counts = agglomerative_cluster(a, 0.5)

    assert np.all(np.isclose(means, [[1.1, 1.466667], [0.15, 0.25], [2.1, 2.2]]))
    assert np.all(np.isclose(stds, [[0.08165, 0.04714], [0.05, 0.05], [0.0, 0.0]]))
    assert np.all(counts == [3, 2, 1])


def test_prepare_data_for_clustering():
    a = np.array(
        [(1.0, 3.0), (2.0, 6.0), (3.0, 9.0)], dtype=[("a", float), ("b", float)]
    )

    pa = prepare_data_for_clustering(a)

    assert np.all(pa == (0.25, 0.75))

    d = {"a": np.arange(5), "b": np.ones(5)}

    pd = prepare_data_for_clustering(d)

    assert np.all(pd[:, 0] == np.arange(5) / (np.arange(5) + 1))
    assert np.all(pd[:, 1] == np.ones(5) / (np.arange(5) + 1))
