import numpy as np
import pytest

from spcal import detection


def test_accumulate_detections():
    x = np.array([0, 1, 3, 1, 0, 2, 4, 2, 0, 4, 2, 6, 2, 0]).astype(float)
    # Lc < Ld
    sums, labels, regions = detection.accumulate_detections(x, 0.5, 1.0)
    assert np.all(sums == [5.0, 8.0, 4.0, 10.0])
    assert np.all(labels == [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 0])
    assert np.all(regions == [[0, 4], [4, 8], [8, 10], [10, 13]])

    # Test regions access
    assert np.all(sums == np.add.reduceat(x, regions.ravel()[:-1])[::2])

    # Lc < Ld, integrate
    sums, labels, regions = detection.accumulate_detections(x, 0.5, 1.0, integrate=True)
    assert np.all(sums == [3.5, 6.5, 3.5, 8.5])

    assert np.all(
        sums == np.add.reduceat(x, regions.ravel()[:-1])[::2] - [1.5, 1.5, 0.5, 1.5]
    )

    # Lc == Ld
    sums, labels, regions = detection.accumulate_detections(x, 1.0, 1.0)
    assert np.all(sums == [4.0, 8.0, 4.0, 10.0])
    assert np.all(labels == [0, 1, 1, 0, 2, 2, 2, 2, 3, 3, 4, 4, 4, 0])
    assert np.all(regions == [[1, 3], [4, 8], [8, 10], [10, 13]])

    # Lc > Ld
    with pytest.raises(ValueError):
        sums, labels, regions = detection.accumulate_detections(x, 1.0, 0.0)

    # Lc > max
    sums, labels, regions = detection.accumulate_detections(x, 7.0, 7.0)
    assert np.all(sums == [])
    assert np.all(labels == np.zeros(x.size, dtype=int))
    assert regions.size == 0

    # Ld > max > Lc
    sums, labels, regions = detection.accumulate_detections(x, 0.0, 7.0)
    assert np.all(sums == [])
    assert np.all(labels == np.zeros(x.size, dtype=int))
    assert regions.size == 0


def test_accumulate_detections_multiple_points():
    x = np.array([0, 3, 0, 0, 3, 3, 0, 0, 3, 3, 3, 0, 0, 0]).astype(float)

    sums, labels, regions = detection.accumulate_detections(
        x, 0.5, 1.0, points_required=2
    )
    assert np.all(sums == [6.0, 9.0])
    assert np.all(labels == [0, 0, 0, 1, 1, 1, 0, 2, 2, 2, 2, 0, 0, 0])

    sums, labels, regions = detection.accumulate_detections(
        x, 0.5, 1.0, points_required=3
    )
    assert np.all(sums == [9.0])
    assert np.all(labels == [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0])

    with pytest.raises(ValueError):
        _, _, _ = detection.accumulate_detections(x, 0.5, 1, points_required=0)


def test_detection_maxima():
    x = np.array([2.0, 1.0, 0.0, 2.0, 3.0, 5.0, 2.0, 3.0, 0.0, 3.0, 0.0])
    regions = np.array([[1, 2], [3, 8], [9, 10]])

    maxima = detection.detection_maxima(x, regions)
    assert np.all(maxima == [1, 5, 9])

    regions = np.array([[1, 2], [3, 8], [6, 7], [9, 10]])

    maxima = detection.detection_maxima(x, regions)
    assert np.all(maxima == [1, 5, 6, 9])


def test_combine_detections():
    # within allowed overlap
    sums = {
        "A": np.array([3.0, 3.0, 3.0]),
        "B": np.array([3.0, 3.0, 3.0]),
        "C": np.array([3.0, 3.0, 3.0]),
    }
    labels = {
        "A": np.array([0, 1, 1, 1, 0, 0, 0, 2, 2, 2, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0]),
        "B": np.array([0, 0, 0, 1, 1, 1, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0]),
        "C": np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0]),
    }
    regions = {
        "A": np.array([[0, 3], [7, 10], [13, 16]]),
        "B": np.array([[2, 5], [7, 10], [16, 19]]),
        "C": np.array([[3, 6], [7, 10], [11, 14]]),
    }

    csums, clabels, cregions = detection.combine_detections(sums, labels, regions)
    assert np.all(csums["A"] == [3.0, 0.0, 0.0, 3.0, 0.0, 3.0, 0.0])
    assert np.all(csums["B"] == [0.0, 3.0, 0.0, 3.0, 0.0, 0.0, 3.0])
    assert np.all(csums["C"] == [0.0, 0.0, 3.0, 3.0, 3.0, 0.0, 0.0])
    assert np.all(
        clabels == [1, 1, 1, 2, 2, 3, 0, 4, 4, 4, 0, 5, 5, 5, 6, 6, 7, 7, 7, 0]
    )
    assert np.all(
        cregions == [[0, 3], [3, 5], [5, 6], [7, 10], [11, 14], [14, 16], [16, 19]]
    )

    sums = {
        "A": np.array([3.0, 2.0]),
        "B": np.array([3.0, 2.0]),
        "C": np.array([3.0, 2.0]),
    }
    labels = {
        "A": np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0]),
        "B": np.array([0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0]),
        "C": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 2, 2, 0]),
    }
    regions = {
        "A": np.array([[0, 3], [14, 16]]),
        "B": np.array([[5, 8], [12, 14]]),
        "C": np.array([[9, 12], [16, 18]]),
    }

    csums, clabels, cregions = detection.combine_detections(sums, labels, regions)
    assert np.all(csums["A"] == [3.0, 0.0, 0.0, 0.0, 2.0, 0.0])
    assert np.all(csums["B"] == [0.0, 3.0, 0.0, 2.0, 0.0, 0.0])
    assert np.all(csums["C"] == [0.0, 0.0, 3.0, 0.0, 0.0, 2.0])

    # sums overlaps
    sums = {"A": np.array([3.0, 3.0]), "B": np.array([3.0, 3.0])}
    labels = {
        "A": np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0]),
        "B": np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, 0]),
    }
    regions = {
        "A": np.array([[0, 6], [6, 14]]),
        "B": np.array([[2, 10], [14, 18]]),
    }

    csums, clabels, cregions = detection.combine_detections(sums, labels, regions)
    assert np.all(csums["A"] == [6.0, 0.0])
    assert np.all(csums["B"] == [3.0, 3.0])
