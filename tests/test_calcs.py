import numpy as np
from numpy.lib import stride_tricks
from nanopart import calc

# Force using custom
calc.bottleneck_found = False


def test_moving_mean():
    x = np.random.random(1000)
    v = stride_tricks.sliding_window_view(x, 9)
    assert np.all(np.isclose(np.mean(v, axis=1), calc.moving_mean(x, 9)))
    v = stride_tricks.sliding_window_view(x, 10)
    assert np.all(np.isclose(np.mean(v, axis=1), calc.moving_mean(x, 10)))


def test_moving_median():
    x = np.random.random(1000)
    v = stride_tricks.sliding_window_view(x, 9)
    assert np.all(np.isclose(np.median(v, axis=1), calc.moving_median(x, 9)))
    v = stride_tricks.sliding_window_view(x, 10)
    assert np.all(np.isclose(np.median(v, axis=1), calc.moving_median(x, 10)))


def test_moving_std():
    x = np.random.random(1000)
    v = stride_tricks.sliding_window_view(x, 9)
    assert np.all(np.isclose(np.std(v, axis=1), calc.moving_std(x, 9)))
    v = stride_tricks.sliding_window_view(x, 10)
    assert np.all(np.isclose(np.std(v, axis=1), calc.moving_std(x, 10)))


def test_calculate_limits():
    x = np.random.poisson(size=100, lam=3.0)

    limits = calc.calculate_limits(x, method="Automatic", sigma=5.0, epsilon=0.5)
    assert limits[0][0] == "Poisson"

    x = np.random.poisson(size=100, lam=60.0)

    limits = calc.calculate_limits(x, method="Automatic", sigma=5.0, epsilon=0.5)
    assert limits[0][0] == "Gaussian"
