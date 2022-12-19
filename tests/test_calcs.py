import numpy as np
import pytest
from numpy.lib import stride_tricks

from spcal import calc, poisson

# Force using custom
calc.BOTTLENECK_FOUND = False


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
