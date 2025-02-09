import numpy as np

from spcal.lib.spcalext.detection import (
    combine_regions,
    label_regions,
    maxima,
    peak_prominence,
)


def test_combine_regions():
    x = np.array([[0, 10], [30, 40], [50, 60]])
    y = np.array([[5, 15], [35, 45], [50, 60]])
    z = np.array([[10, 20], [58, 62]])

    c = combine_regions([x, y, z], 1)
    assert np.all(c == [[0, 20], [30, 45], [50, 62]])

    c = combine_regions([x, y, z], 3)
    assert np.all(c == [[0, 20], [30, 45], [50, 60], [58, 62]])


def test_label_regions():
    r = np.array([[0, 4], [6, 8], [8, 9]])
    labels = label_regions(r, 10)
    assert np.all(labels == [1, 1, 1, 1, 0, 0, 2, 2, 3, 0])


def test_peak_prominence():
    y = np.array(
        [0, 0, 0, 1, 2, 3, 2, 3, 1, 3, 5, 6, 5, 3, 2, 3, 1, 0, 0, 0], dtype=float
    )
    prom, left, right = peak_prominence(y, np.array([5, 11]), 0.5)
    assert np.allclose(prom, [2.0, 6.0])
    assert np.all(left == [2, 2])
    assert np.all(right == [8, 17])

    y = np.array(
        [0, 0, 0, 1, 0, 0, 7, 1, 0, 0, 1, 0, 0, 2, 3, 3, 1, 1, 2, 1], dtype=float
    )
    prom, left, right = peak_prominence(y, np.array([6, 14]), 0.5)
    assert np.allclose(prom, [7.0, 2.0])
    assert np.all(left == [5, 12])
    assert np.all(right == [8, 19])


def test_maxima():
    a = np.arange(10.0)
    b = np.array([[1, 3], [2, 4], [3, 7], [9, 9]])
    m = maxima(a, b)
    assert np.all(m == [2, 3, 6, 9])
