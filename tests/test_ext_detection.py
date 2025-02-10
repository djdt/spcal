import numpy as np

from spcal.lib.spcalext.detection import (
    combine_regions,
    label_regions,
    maxima,
    peak_prominence,
)


def test_combine_regions():
    x = np.array([[0, 10], [30, 40], [50, 60], [60, 70]])
    y = np.array([[5, 15], [35, 45], [50, 61]])
    z = np.array([[10, 20], [56, 58]])

    c = combine_regions([x, y, z], 0)
    assert np.all(c == [[0, 20], [30, 45], [50, 70]])

    c = combine_regions([x, y, z], 1)
    assert np.all(c == [[0, 20], [30, 45], [50, 61], [60, 70]])


def test_label_regions():
    r = np.array([[0, 4], [6, 8], [8, 9]])
    labels = label_regions(r, 10)
    assert np.all(labels == [1, 1, 1, 1, 0, 0, 2, 2, 3, 0])


def test_peak_prominence():
    # separate
    y = np.array(  # ,  |  v     |           |     v     |
        [0, 0, 0, 1, 0, 0, 7, 1, 0, 0, 1, 0, 0, 2, 3, 3, 1, 1, 2, 1], dtype=float
    )
    prom, left, right = peak_prominence(y, np.array([6, 14]))
    assert np.allclose(prom, [7.0, 2.0])
    assert np.all(left == [5, 12])
    assert np.all(right == [8, 16])

    # joined
    y = np.array(  # ,  |  v                    |  v     |
        [0, 0, 0, 1, 0, 0, 7, 6, 4, 3, 5, 6, 5, 2, 3, 3, 1, 1, 2, 1], dtype=float
    )
    prom, left, right = peak_prominence(y, np.array([6, 14]))
    assert np.allclose(prom, [6.0, 1.0])
    assert np.all(left == [5, 13])
    assert np.all(right == [16, 16])


def test_peak_prominence_dropped_peaks():
    y = np.full((1000, 3), 0.01, dtype=np.float32)
    y[5::10, 0] += 100 + np.arange(0, 100)
    y[5::10, 1] += 100 + np.arange(0, 100) * 2
    y[5::10, 2] += 100 + np.tile([10, 20, 30, 40, 50], 20)

    for ax in [0, 1, 2]:
        prom, left, right = peak_prominence(y[:, ax], np.arange(5, 1000, 10))
        assert prom.size == 100
    assert False


def test_maxima():
    a = np.arange(10.0)
    b = np.array([[1, 3], [2, 4], [3, 7], [9, 9]])
    m = maxima(a, b)
    assert np.all(m == [2, 3, 6, 9])
