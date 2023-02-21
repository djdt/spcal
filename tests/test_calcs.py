import numpy as np
from numpy.lib import stride_tricks

from spcal import calc

# Force using custom
calc.BOTTLENECK_FOUND = False


def test_moving_mean():
    x = np.random.random(1000)
    v = stride_tricks.sliding_window_view(x, 9)
    assert np.allclose(np.mean(v, axis=1), calc.moving_mean(x, 9))
    v = stride_tricks.sliding_window_view(x, 10)
    assert np.allclose(np.mean(v, axis=1), calc.moving_mean(x, 10))


def test_moving_median():
    x = np.random.random(1000)
    v = stride_tricks.sliding_window_view(x, 9)
    assert np.allclose(np.median(v, axis=1), calc.moving_median(x, 9))
    v = stride_tricks.sliding_window_view(x, 10)
    assert np.allclose(np.median(v, axis=1), calc.moving_median(x, 10))


def test_moving_std():
    x = np.random.random(1000)
    v = stride_tricks.sliding_window_view(x, 9)
    assert np.allclose(np.std(v, axis=1), calc.moving_std(x, 9))
    v = stride_tricks.sliding_window_view(x, 10)
    assert np.allclose(np.std(v, axis=1), calc.moving_std(x, 10))


def test_otsu():
    x = np.cos(np.linspace(0, np.pi, 1000, endpoint=True))
    t = calc.otsu(x, nbins=256)
    assert t == -0.00390625  # skimage.filters.threshold_otsu(x)


def test_pca():
    x = np.sin(np.arange(16).reshape(4, 4))
    a, v, exv = calc.pca(x, trim_to_components=4)

    # Values are from sklearn.decomposition.PCA (Standard scaled)
    assert np.allclose(
        exv, [6.09368614e-01, 3.90631386e-01, 4.84106293e-33, 2.03935216e-36]
    )
    assert np.allclose(
        a,
        [
            [3.02333159e-02, -1.35335306e00, 2.18613315e-16, 1.71224475e-18],
            [-1.39018512e00, 1.78299269e00, 9.31071203e-17, 2.43169507e-18],
            [2.53626718e00, 5.42537306e-01, -1.82419253e-17, 3.06999110e-18],
            [-1.17631538e00, -9.72176943e-01, -1.43748120e-16, 3.78944142e-18],
        ],
    )
