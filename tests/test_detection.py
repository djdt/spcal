from pathlib import Path

import numpy as np
import pytest

from spcal import detection, poisson


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

    # Lc < Ld, integrate
    sums, regions = detection.accumulate_detections(x, 0.5, 1.0, integrate=True)
    assert np.all(sums == [3.5, 6.5, 3.5, 8.5])

    assert np.all(
        sums == np.add.reduceat(x, regions.ravel()[:-1])[::2] - [1.5, 1.5, 0.5, 1.5]
    )

    # Lc == Ld
    sums, regions = detection.accumulate_detections(x, 1.0, 1.0)
    assert np.all(sums == [4.0, 8.0, 4.0, 10.0])
    assert np.all(regions == [[1, 3], [4, 8], [8, 10], [10, 13]])

    # Lc > Ld
    with pytest.raises(ValueError):
        sums, regions = detection.accumulate_detections(x, 1.0, 0.0)

    # Lc > max
    sums, regions = detection.accumulate_detections(x, 7.0, 7.0)
    assert np.all(sums == [])
    assert regions.size == 0

    # Ld > max > Lc
    sums, regions = detection.accumulate_detections(x, 0.0, 7.0)
    assert np.all(sums == [])
    assert regions.size == 0


def test_accumulate_detections_edges():
    x = np.array([9, 1, 3, 1, 0, 2, 4, 2, 0, 4, 2, 6, 2, 1]).astype(float)
    sums, regions = detection.accumulate_detections(
        x, 0.5, 1.0, prominence_required=0.0
    )
    assert np.all(sums == [5.0, 8.0, 4.0, 10.0])


def test_accumulate_detections_windowed():
    x = np.array([0, 1, 3, 1, 0, 2, 4, 2, 0, 4, 2, 6, 2, 0]).astype(float)
    lc = np.full(x.shape, 0.5)
    ld = np.array(
        [1.0, 2.0, 2.0, 2.0, 1.0, 4.0, 6.0, 6.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0]
    )
    sums, regions = detection.accumulate_detections(x, lc, ld, prominence_required=0.0)
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
    sums, regions = detection.accumulate_detections(
        x, 0.5, 1.0, prominence_required=0.0
    )
    assert np.all(sums == [16.0, 16.0])


def test_background_mask():
    mask = detection.background_mask(np.array([[3, 5], [6, 8]]), 10)
    assert np.all(mask[:3])
    assert not np.any(mask[3:5])
    assert np.all(mask[5])
    assert not np.any(mask[6:8])
    assert np.all(mask[8:])


def test_detection_maxima():
    x = np.array([2.0, 1.0, 0.0, 2.0, 3.0, 5.0, 2.0, 3.0, 0.0, 3.0, 0.0])
    regions = np.array([[1, 2], [3, 8], [9, 10]])

    maxima = detection.detection_maxima(x, regions)
    assert np.all(maxima == [1, 5, 9])

    regions = np.array([[1, 2], [3, 8], [6, 7], [9, 10]])

    maxima = detection.detection_maxima(x, regions)
    assert np.all(maxima == [1, 5, 6, 9])


def test_single_particle_peak_splitting():
    path = Path(__file__).parent.joinpath("data/ti_split_peaks.npz")
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


def test_noise_level():
    """Tests detections / integrations stay the same when noise levels rise.
    Going further beyond the limit of crit will start to integrate noise.
    """
    data = np.load(Path(__file__).parent.joinpath("data/agilent_au_data.npz"))
    x = data["au50nm"]
    ub = np.mean(x)
    yc, _ = poisson.formula_c(ub, alpha=0.001)
    detections, _ = detection.accumulate_detections(x, ub, yc + ub, integrate=True)

    truth = detections.size
    truth_mean = detections.mean()

    np.random.seed(14234)

    for lam in np.linspace(ub, yc, 5, endpoint=True):
        noise = np.random.poisson(lam=lam, size=x.size)

        noise_x = x + noise

        noise_ub = np.mean(noise_x)
        noise_yc, _ = poisson.formula_c(noise_ub, alpha=0.001)
        detections, regions = detection.accumulate_detections(
            noise_x,
            noise_ub,
            noise_yc + noise_ub,
            integrate=True,
        )

        # Less than 1% change in number of detections
        assert abs(truth - detections.size) / truth < 0.01
        # Less than 5% change in mean detected area
        assert abs(truth_mean - detections.mean()) / truth_mean < 0.05
