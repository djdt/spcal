import numpy as np
import pytest
from nanopart.util import accumulate_detections


def test_accumulate_detections():
    x = np.array([2, 1, 2, 2, 1, 0, 0, 1, 0, 2])
    # Lc == Ld
    sums, labels = accumulate_detections(x, 1, 1)
    assert np.all(sums == [2, 4, 2])
    assert np.all(labels == [1, 0, 2, 2, 0, 0, 0, 0, 0, 3])

    # Lc < Ld
    sums, labels = accumulate_detections(x, 0, 1)
    assert np.all(sums == [8, 2])
    assert np.all(labels == [1, 1, 1, 1, 1, 0, 0, 0, 0, 2])

    # Lc > Ld
    with pytest.raises(ValueError):
        sums, labels = accumulate_detections(x, 1, 0)

    # Lc > max
    sums, labels = accumulate_detections(x, 3, 3)
    assert np.all(sums == [])
    assert np.all(labels == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # Ld > max > Lc
    sums, labels = accumulate_detections(x, 0, 3)
    assert np.all(sums == [])
    assert np.all(labels == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
