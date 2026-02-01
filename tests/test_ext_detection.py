import numpy as np
from scipy.signal import peak_prominences

from spcal.lib.spcalext import detection


def test_local_maxima():
    a = np.zeros(30)
    r = np.array([0, 5, 7, 19, 20, 21, 25])
    a[r] += 1
    m = detection.local_maxima(a)
    assert np.all(np.flatnonzero(m) == [0, 5, 7, 19, 25])


def test_maxima_between():
    a = np.arange(10.0)
    b = np.array([[1, 3], [2, 4], [3, 7], [9, 9]])
    m = detection.maxima_between(a, b)
    assert np.all(m == [2, 3, 6, 9])


def test_peak_prominence():
    # separate
    y = np.array(  # ,  |  v     |           |     v     |
        [0, 0, 0, 1, 0, 0, 7, 1, 0, 0, 1, 0, 0, 2, 3, 3, 1, 1, 2, 1], dtype=float
    )
    prom, left, right = detection.peak_prominence(y, np.array([6, 14]))
    assert np.allclose(prom, [7.0, 2.0])
    assert np.all(left == [5, 12])
    assert np.all(right == [8, 16])

    # joined
    y = np.array(  # ,  |  v                    |  v     |
        [0, 0, 0, 1, 0, 0, 7, 6, 4, 3, 5, 6, 5, 2, 3, 3, 1, 1, 2, 1], dtype=float
    )
    prom, left, right = detection.peak_prominence(y, np.array([6, 14]))
    assert np.allclose(prom, [6.0, 1.0])
    assert np.all(left == [5, 13])
    assert np.all(right == [16, 16])

    # no min
    y = np.array(
        [0, 0, 1, 1, 5, 9, 7, 3, 2, 1, 1, 1, 1, 0, 1, 1, 1, 2, 3, 1], dtype=float
    )
    prom, left, right = detection.peak_prominence(y, np.array([5]))
    assert np.allclose(prom, [9.0])
    assert np.all(left == [1])
    assert np.all(right == [13])

    # min=1
    prom, left, right = detection.peak_prominence(y, np.array([5]), min_base=1.0)
    assert np.allclose(prom, [8.0])
    assert np.all(left == [3])
    assert np.all(right == [9])

    # min=array
    min = np.array(
        [0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], dtype=float
    )
    prom, left, right = detection.peak_prominence(y, np.array([5]), min_base=min)
    assert np.allclose(prom, [6.0])
    assert np.all(left == [1])
    assert np.all(right == [7])


def test_peak_prominence_dropped_peaks():
    y = np.full((1000, 3), 0.01, dtype=np.float32)
    y[5::10, 0] += 100 + np.arange(0, 100)
    y[5::10, 1] += 100 + np.arange(0, 100) * 2
    y[5::10, 2] += 100 + np.tile([10, 20, 30, 40, 50], 20)

    for ax in [0, 1, 2]:
        prom, left, right = detection.peak_prominence(y[:, ax], np.arange(5, 1000, 10))
        assert prom.size == 100


def test_peak_prominence_scipy():
    x = np.random.random(100)
    y = np.random.choice(100, 10)

    p1, l1, r1 = detection.peak_prominence(x, y)
    p2, l2, r2 = peak_prominences(x, y)

    assert np.allclose(p1, p2)
    assert np.all(l1 == l2)
    assert np.all(r1 == r2)


def test_label_regions():
    r = np.array([[0, 4], [6, 8], [8, 9]])
    labels = detection.label_regions(r, 10)
    assert np.all(labels == [1, 1, 1, 1, 0, 0, 2, 2, 3, 0])

    r = np.array([[0, 4], [3, 5], [8, 9]])
    labels = detection.label_regions(r, 10)
    assert np.all(labels == [1, 1, 1, 2, 2, 0, 0, 0, 3, 0])


def test_combine_regions():
    # singular
    a = np.array([[0, 10]])
    b = np.array([[20, 30]])
    c = np.array([[40, 50]])
    d = detection.combine_regions([a, b, c])
    assert np.all(d == [[0, 10], [20, 30], [40, 50]])
    d = detection.combine_regions([a, b, c], 5)
    assert np.all(d == [[0, 10], [20, 30], [40, 50]])
    d = detection.combine_regions([a, b, c], 15)
    assert np.all(d == [[0, 10], [20, 30], [40, 50]])

    # overlapping singular
    a = np.array([[0, 10], [20, 30], [40, 50]])
    b = np.array([[0, 10], [20, 30], [40, 50]])
    c = np.array([[0, 10], [20, 30], [40, 50]])
    d = detection.combine_regions([a, b, c])
    assert np.all(d == a)
    d = detection.combine_regions([a, b, c], 5)
    assert np.all(d == a)
    d = detection.combine_regions([a, b, c], 15)
    assert np.all(d == a)

    # overlapping 1:2
    a = np.array([[0, 20]])
    b = np.array([[10, 30]])
    c = np.array([[40, 50]])
    d = detection.combine_regions([a, b, c])
    assert np.all(d == [[0, 30], [40, 50]])
    d = detection.combine_regions([a, b, c], 5)
    assert np.all(d == [[0, 30], [40, 50]])
    d = detection.combine_regions([a, b, c], 15)
    assert np.all(d == [[0, 16], [16, 30], [40, 50]])

    # overlapping 1:2:3
    a = np.array([[0, 20]])
    b = np.array([[10, 30]])
    c = np.array([[20, 50]])
    d = detection.combine_regions([a, b, c])
    assert np.all(d == [[0, 50]])
    d = detection.combine_regions([a, b, c], 5)
    assert np.all(d == [[0, 50]])
    d = detection.combine_regions([a, b, c], 15)
    assert np.all(d == [[0, 16], [16, 26], [26, 50]])

    # overlapping 2:3
    a = np.array([[0, 10]])
    b = np.array([[20, 40]])
    c = np.array([[30, 50]])
    d = detection.combine_regions([a, b, c])
    assert np.all(d == [[0, 10], [20, 50]])
    d = detection.combine_regions([a, b, c], 5)
    assert np.all(d == [[0, 10], [20, 50]])
    d = detection.combine_regions([a, b, c], 15)
    assert np.all(d == [[0, 10], [20, 36], [36, 50]])

    # inside 1:2
    a = np.array([[0, 30]])
    b = np.array([[10, 20]])
    c = np.array([[40, 50]])
    d = detection.combine_regions([a, b, c])
    assert np.all(d == [[0, 30], [40, 50]])
    d = detection.combine_regions([a, b, c], 5)
    assert np.all(d == [[0, 30], [40, 50]])
    d = detection.combine_regions([a, b, c], 15)
    assert np.all(d == [[0, 30], [40, 50]])

    # inside 1:2:3
    a = np.array([[2, 30]])
    b = np.array([[10, 40]])
    c = np.array([[0, 50]])
    d = detection.combine_regions([a, b, c])
    assert np.all(d == c)
    d = detection.combine_regions([a, b, c], 5)
    assert np.all(d == c)
    d = detection.combine_regions([a, b, c], 15)
    assert np.all(d == c)

    # inside 1:2 overlapping 3
    a = np.array([[0, 40]])
    b = np.array([[10, 30]])
    c = np.array([[20, 50]])
    d = detection.combine_regions([a, b, c])
    assert np.all(d == [[0, 50]])
    d = detection.combine_regions([a, b, c], 5)
    assert np.all(d == [[0, 50]])
    d = detection.combine_regions([a, b, c], 15)
    assert np.all(d == [[0, 50]])

    # inside 1:2:3 overlapping 2:3
    a = np.array([[0, 50]])
    b = np.array([[10, 30]])
    c = np.array([[20, 40]])
    d = detection.combine_regions([a, b, c])
    assert np.all(d == [[0, 50]])
    d = detection.combine_regions([a, b, c], 5)
    assert np.all(d == [[0, 50]])
    d = detection.combine_regions([a, b, c], 15)
    assert np.all(d == [[0, 50]])

    # overlapping 1:2 inside 2:3
    a = np.array([[0, 30]])
    b = np.array([[20, 50]])
    c = np.array([[30, 40]])
    d = detection.combine_regions([a, b, c])
    assert np.all(d == [[0, 50]])
    d = detection.combine_regions([a, b, c], 5)
    assert np.all(d == [[0, 50]])
    d = detection.combine_regions([a, b, c], 15)
    assert np.all(d == [[0, 26], [26, 50]])

    # touching
    a = np.array([[0, 20]])
    b = np.array([[20, 30]])
    c = np.array([[30, 50]])
    d = detection.combine_regions([a, b, c])
    assert np.all(d == [[0, 20], [20, 30], [30, 50]])
    d = detection.combine_regions([a, b, c], 5)
    assert np.all(d == [[0, 20], [20, 30], [30, 50]])
    d = detection.combine_regions([a, b, c], 15)
    assert np.all(d == [[0, 20], [20, 30], [30, 50]])

    # negative width error
    a = np.array([[0, 50]])
    b = np.array([[20, 30], [30, 40]])
    c = np.array([[40, 50]])
    d = detection.combine_regions([a, b, c])
    assert np.all(d == [[0, 50]])


def test_split_peaks():
    # single peak
    prom, left, right = [6.0], [4], [6]
    _left, _right = detection.split_peaks(prom, left, right, 0.5)
    assert _left == left
    assert _right == right

    # two peaks left larger, split
    prom, left, right = [7.0, 5.0], [4, 7], [9, 9]
    _left, _right = detection.split_peaks(prom, left, right, 0.2)
    assert np.all(_left == left)
    assert np.all(_right == [7, 9])

    # two peaks left larger, merge
    _left, _right = detection.split_peaks(prom, left, right, 1.0)
    assert np.all(_left == [4])
    assert np.all(_right == [9])

    # two peaks right larger, split
    prom, left, right = [5.0, 7.0], [4, 7], [9, 9]
    _left, _right = detection.split_peaks(prom, left, right, 0.2)
    assert np.all(_left == left)
    assert np.all(_right == [7, 9])

    # two peaks right larger, merge
    _left, _right = detection.split_peaks(prom, left, right, 1.0)
    assert np.all(_left == [4])
    assert np.all(_right == [9])

    # three peaks, mid largest, split
    prom, left, right = detection.peak_prominence(
        np.array([0, 0, 0, 1, 2, 1, 3, 5, 2, 3, 0, 0]), np.array([4, 7, 9])
    )
    _left, _right = detection.split_peaks(prom, left, right, 0.1)
    assert np.all(_left == [2, 5, 8])
    assert np.all(_right == [5, 8, 10])

    # three peaks, mid largest, merge
    _left, _right = detection.split_peaks(prom, left, right, 1.0)
    assert np.all(_left == [2])
    assert np.all(_right == [10])

    # previous error
    prom, left, right = [3.0, 4.0, 2.0, 6.0], [0, 4, 8, 8], [4, 8, 10, 13]
    _left, _right = detection.split_peaks(prom, left, right, 0.0)
    assert np.all(_left == [0, 4, 8, 10])
    assert np.all(_right == [4, 8, 10, 13])

    _left, _right = detection.split_peaks(prom, left, right, 1.0)
    assert np.all(_left == [0, 4, 8])
    assert np.all(_right == [4, 8, 13])
