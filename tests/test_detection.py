import numpy as np
import pytest

from spcal import detection


def test_accumulate_detections():
    x = np.array([2, 1, 2, 2, 1, 0, 0, 1, 0, 2])
    # Lc == Ld
    sums, labels, regions = detection.accumulate_detections(x, 1, 1)
    assert np.all(sums == [2, 4, 2])
    assert np.all(labels == [1, 0, 2, 2, 0, 0, 0, 0, 0, 3])
    assert np.all(regions == [[0, 1], [2, 4], [9, 9]])

    # Test regions access
    assert np.all(sums == np.add.reduceat(x, regions.ravel())[::2])

    # Lc < Ld
    sums, labels, regions = detection.accumulate_detections(x, 0, 1)
    assert np.all(sums == [8, 2])
    assert np.all(labels == [1, 1, 1, 1, 1, 0, 0, 0, 0, 2])
    assert np.all(regions == [[0, 5], [9, 9]])

    # Lc > Ld
    with pytest.raises(ValueError):
        sums, labels, regions = detection.accumulate_detections(x, 1, 0)

    # Lc > max
    sums, labels, regions = detection.accumulate_detections(x, 3, 3)
    assert np.all(sums == [])
    assert np.all(labels == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert regions.size == 0

    # Ld > max > Lc
    sums, labels, regions = detection.accumulate_detections(x, 0, 3)
    assert np.all(sums == [])
    assert np.all(labels == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert regions.size == 0


def test_detection_maxima():
    x = np.array([2.0, 1.0, 0.0, 2.0, 3.0, 5.0, 2.0, 3.0, 0.0, 3.0, 0.0])
    regions = np.array([[1, 2], [3, 8], [9, 10]])

    maxima = detection.detection_maxima(x, regions)
    assert np.all(maxima == [1, 5, 9])

    regions = np.array([[1, 2], [3, 8], [6, 7], [9, 10]])

    maxima = detection.detection_maxima(x, regions)
    assert np.all(maxima == [1, 5, 6, 9])


def test_combine_detections():
    sums = {
        "A": np.array([1.0, 2.0, 4.0, 8.0]),
        "B": np.array([8.0, 4.0, 2.0, 8.0]),
    }
    labels = {
        "A": np.array([0, 1, 1, 0, 2, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 4, 4, 0, 0]),
        "B": np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 2, 2, 0, 3, 3, 0, 0, 4, 4, 0]),
    }
    regions = {
        "A": np.array([[1, 3], [4, 6], [11, 12], [15, 17]]),
        "B": np.array([[2, 5], [9, 11], [12, 14], [16, 18]]),
    }

    csums, clabels, cregions = detection.combine_detections(sums, labels, regions)
    assert np.all(csums["A"] == [3.0, 4.0, 8.0])
    assert np.all(csums["B"] == [8.0, 6.0, 8.0])
    assert np.all(clabels == [0, 1, 1, 1, 1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 0, 3, 3, 3, 0])
    assert np.all(cregions == [[1, 6], [9, 14], [15, 18]])

    # Start at start, end at array end
    labels = {
        "A": np.array([1, 1, 1, 0, 2, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 4, 4, 0, 0]),
        "B": np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 2, 2, 0, 3, 3, 0, 0, 4, 4, 4]),
    }
    csums, clabels, cregions = detection.combine_detections(sums, labels, regions)
    assert np.all(csums["A"] == [3.0, 4.0, 8.0])
    assert np.all(csums["B"] == [8.0, 6.0, 8.0])
    assert np.all(clabels == [1, 1, 1, 1, 1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 0, 3, 3, 3, 3])
    assert np.all(cregions == [[0, 6], [9, 14], [15, 18]])
