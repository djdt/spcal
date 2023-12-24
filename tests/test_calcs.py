import numpy as np
import pytest
from scipy.special import erf as erf_sp
from scipy.special import erfinv as erfinv_sp

from spcal import calc


def test_is_integer_or_near():
    assert calc.is_integer_or_near(1.05, max_deviation=0.1)
    assert calc.is_integer_or_near(0.95, max_deviation=0.1)
    assert calc.is_integer_or_near(100.01, max_deviation=0.1)
    assert not calc.is_integer_or_near(1.25, max_deviation=0.1)

    assert np.all(
        calc.is_integer_or_near(np.array([0.96, 1.97, 2.98, 3.99]), max_deviation=0.1)
    )

    with pytest.raises(ValueError):
        calc.is_integer_or_near(1.0, max_deviation=-0.1)


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


def test_weighting():
    # Zero length x should return zero length weights
    assert calc.weights_from_weighting(np.array([]), "x").size == 0
    # Test all the weights, safe return
    x = np.random.normal(loc=2, size=10)
    assert np.all(calc.weights_from_weighting(x, "Equal") == 1.0)
    assert np.all(calc.weights_from_weighting(x, "x") == x)
    assert np.all(calc.weights_from_weighting(x, "1/x") == 1 / x)
    assert np.all(calc.weights_from_weighting(x, "1/(x^2)") == 1 / (x * x))

    # Test safe return
    x[0] = 0.0
    assert calc.weights_from_weighting(x, "x", True)[0] == np.amin(x[1:])
    assert np.all(calc.weights_from_weighting(x, "x", False) == x)
    with pytest.raises(ValueError):
        calc.weights_from_weighting(x, "invalid")

    # Nan ignored when looking for min value in safe
    assert np.all(
        calc.weights_from_weighting(np.array([0.0, 1.0, np.nan, 2.0]), "1/x", True)[
            [0, 1, 3]
        ]
        == [1.0, 1.0, 0.5]
    )

    # Reuturn all ones if x all 0
    assert np.all(calc.weights_from_weighting(np.zeros(10), "1/x") == 1.0)


def test_weighted_rsq():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 1.0, 2.0, 4.0, 8.0])
    # From http://www.xuru.org/rt/WLR.asp
    assert calc.weighted_rsq(x, y, None) == pytest.approx(8.30459e-1)
    assert calc.weighted_rsq(x, y, x) == pytest.approx(8.65097e-1)
    assert calc.weighted_rsq(x, y, 1 / x) == pytest.approx(7.88696e-1)
    assert calc.weighted_rsq(x, y, 1 / (x**2)) == pytest.approx(7.22560e-1)


def test_weighted_linreg():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 1.0, 2.0, 4.0, 8.0])
    # Normal
    assert calc.weighted_linreg(x, y, None) == pytest.approx(
        (1.7, -1.9, 0.830459, 1.402379)
    )
    # Weighted
    assert calc.weighted_linreg(x, y, x) == pytest.approx(
        (2.085714, -3.314286, 0.865097, 1.603991)
    )
