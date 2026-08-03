from pathlib import Path

import numpy as np
import pytest

from spcal import detection


def test_accumulate_detections():
    x = np.array([0, 1, 3, 1, 0, 2, 4, 2, 0, 4, 2, 6, 2, 0]).astype(float)
    # Lc < Ld
    sums, regions = detection.accumulate_detections(
        x, 0.5, 1.0, prominence_required=0.0
    )
    assert np.all(sums == [5.0, 8.0, 4.0, 10])
    assert np.all(regions == [[0, 4], [4, 8], [8, 10], [10, 13]])

    # Test regions access
    assert np.all(sums == np.add.reduceat(x, regions.ravel()[:-1])[::2])

    # Lc == Ld
    sums, regions = detection.accumulate_detections(x, 1.0, 1.0)
    assert np.all(sums == [4.0, 8.0, 14.0])
    assert np.all(regions == [[1, 3], [4, 8], [8, 13]])

    # Lc == Ld, prominence at 0.2
    sums, regions = detection.accumulate_detections(
        x, 1.0, 1.0, prominence_required=0.2
    )
    assert np.all(sums == [4.0, 8.0, 4.0, 10.0])
    assert np.all(regions == [[1, 3], [4, 8], [8, 10], [10, 13]])

    # Lc > Ld
    with pytest.raises(ValueError):
        detection.accumulate_detections(x, 1.0, 0.0)

    # Lc > max
    sums, regions = detection.accumulate_detections(x, 7.0, 7.0)
    assert np.all(sums == [])
    assert regions.size == 0

    # Ld > max > Lc
    sums, regions = detection.accumulate_detections(x, 0.0, 7.0)
    assert np.all(sums == [])
    assert regions.size == 0


def test_accumulate_detections_zeros():
    x = np.zeros(51, dtype=np.float32)

    sums, regions = detection.accumulate_detections(
        x, 0.0, 1.0, prominence_required=0.0
    )
    assert len(sums) == 0
    assert len(regions) == 0

    sums, regions = detection.accumulate_detections(
        x, 0.0, 1.0, prominence_required=0.2
    )
    assert len(sums) == 0
    assert len(regions) == 0


def test_accumulate_detections_edges():
    x = np.array([9, 1, 3, 1, 0, 2, 4, 2, 0, 4, 2, 6, 2, 1]).astype(float)
    sums, _ = detection.accumulate_detections(x, 0.5, 1.0, prominence_required=0.0)
    assert np.all(sums == [5.0, 8.0, 4.0, 10.0])


def test_accumulate_detections_windowed():
    x = np.array([0, 1, 3, 1, 0, 2, 4, 2, 0, 4, 2, 6, 2, 0]).astype(float)
    lc = np.full(x.shape, 0.5)
    ld = np.array(
        [1.0, 2.0, 2.0, 2.0, 1.0, 4.0, 6.0, 6.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0]
    )
    sums, _ = detection.accumulate_detections(x, lc, ld, prominence_required=0.0)
    assert np.all(sums == [5.0, 4.0, 10.0])


def test_accumulate_detections_multiple_points():
    x = np.array([0, 3, 0, 0, 3, 3, 0, 0, 3, 3, 3, 0, 0, 0]).astype(float)

    sums, regions = detection.accumulate_detections(x, 0.5, 1.0, points_required=2)
    assert np.all(sums == [6.0, 9.0])
    assert np.all(regions == [[3, 6], [7, 11]])

    sums, regions = detection.accumulate_detections(x, 0.5, 1.0, points_required=3)
    assert np.all(sums == [9.0])
    assert np.all(regions == [[7, 11]])

    with pytest.raises(ValueError):
        _, _ = detection.accumulate_detections(x, 0.5, 1, points_required=0)


def test_accumulate_detections_prominence():
    x = np.array([0, 0, 5, 2, 6, 2, 0, 0, 5, 4, 3, 2, 0, 0, 5, 4, 6, 0]).astype(float)

    sums, regions = detection.accumulate_detections(
        x, 0.5, 1.0, prominence_required=0.0
    )
    assert np.all(sums == [5.0, 10.0, 14.0, 5.0, 10.0])
    assert np.all(regions == [[1, 3], [3, 6], [7, 12], [13, 15], [15, 17]])

    sums, regions = detection.accumulate_detections(
        x, 0.5, 1.0, prominence_required=0.2
    )
    assert np.all(sums == [5.0, 10.0, 14.0, 15.0])
    assert np.all(regions == [[1, 3], [3, 6], [7, 12], [13, 17]])

    sums, regions = detection.accumulate_detections(
        x, 0.5, 1.0, prominence_required=1.0
    )
    assert np.all(sums == [15.0, 14.0, 15.0])
    assert np.all(regions == [[1, 6], [7, 12], [13, 17]])

    # strange case with same max peak
    x = np.array([0, 0, 3, 5, 2, 5, 1, 0, 0, 0, 5, 6, 5, 0]).astype(float)
    sums, _ = detection.accumulate_detections(x, 0.5, 1.0, prominence_required=0.0)
    assert np.all(sums == [16.0, 16.0])


def test_background_mask():
    mask = detection.background_mask(np.array([[3, 5], [6, 8]]), 10)
    assert np.all(mask[:3])
    assert not np.any(mask[3:5])
    assert np.all(mask[5])
    assert not np.any(mask[6:8])
    assert np.all(mask[8:])


def test_detection_baselines():
    x = np.array([1.0, 2.0, 5.0, 2.0, 1.0, 10.0, 20.0, 5.0, 1.0])
    regions = np.array([[2, 3], [5, 8]])

    bases = detection.detection_baselines(x, regions)
    assert np.all(bases == [5.0, 35.0])

    bases = detection.detection_baselines(1.0, regions)
    assert np.all(bases == [1.0, 3.0])


def test_detection_maxima():
    x = np.array([2.0, 1.0, 0.0, 2.0, 3.0, 5.0, 2.0, 3.0, 0.0, 3.0, 0.0])
    regions = np.array([[1, 2], [3, 8], [9, 10]])

    maxima = detection.detection_maxima(x, regions)
    assert np.all(maxima == [1, 5, 9])

    regions = np.array([[1, 2], [3, 8], [6, 7], [9, 10]])

    maxima = detection.detection_maxima(x, regions)
    assert np.all(maxima == [1, 5, 6, 9])


def test_single_particle_peak_splitting(test_data_path: Path):
    path = test_data_path.joinpath("ti_split_peaks.npz")
    x = np.load(path)
    loa, lod = 19.90, 45.0

    # Split peak, largest on left
    sums, regions = detection.accumulate_detections(
        x["a"], loa, lod, prominence_required=0.2, points_required=1
    )
    assert sums.size == 2
    assert regions[0][1] == regions[1][0]

    # Two peaks, with many maxima
    sums, regions = detection.accumulate_detections(
        x["b"], loa, lod, prominence_required=0.2, points_required=1
    )
    assert sums.size == 2
    assert regions[0][1] < regions[1][0]

    # Split peak, largest on right, another peak
    sums, regions = detection.accumulate_detections(
        x["c"], loa, lod, prominence_required=0.2, points_required=1
    )
    assert sums.size == 3
    assert regions[0][1] == regions[1][0]


def test_single_particle_prominence_required(test_data_path: Path):
    """Failing prominence required"""
    path = test_data_path.joinpath("au_single_peak_prominence.npz")
    x = np.load(path)["au197"]
    loa, lod = 0.737286, 12.2188

    sums, _ = detection.accumulate_detections(
        x, loa, lod, prominence_required=0.0, points_required=1
    )
    assert sums.size > 1
    sums, _ = detection.accumulate_detections(
        x, loa, lod, prominence_required=0.1, points_required=1
    )
    assert sums.size == 1
