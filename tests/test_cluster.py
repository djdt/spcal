import numpy as np
from spcal.cluster import agglomerative_cluster


def test_agglomerative_cluster():
    a = np.array(
        [[0.1, 0.2], [0.2, 0.3], [1.0, 1.4], [1.1, 1.5], [1.2, 1.5], [2.1, 2.2]]
    )

    means, counts = agglomerative_cluster(a, 0.5)

    assert np.all(np.isclose(means, [[1.1, 1.466667], [0.15, 0.25], [2.1, 2.2]]))
    assert np.all(counts == [3, 2, 1])
