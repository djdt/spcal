import numpy as np
from numpy.lib import stride_tricks
from spcal import calc

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


def test_calculate_limits_automatic():
    for lam in np.linspace(1.0, 100.0, 25):
        x = np.random.poisson(size=1000, lam=lam)
        limits = calc.calculate_limits(x, method="Automatic", sigma=3.0, error_rates=(0.05, 0.05))
        assert limits[0] == ("Poisson" if lam < 50.0 else "Gaussian")
