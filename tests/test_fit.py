import numpy as np

from spcal import fit
from spcal.dists import lognormal, normal


def test_neadler_mean():
    def gradient(x: np.ndarray, y: np.ndarray, a: float, b: float) -> float:
        return np.sum(np.abs((x * a + np.log(b)) - y))

    x = np.arange(100.0)
    y = x * 3.0 + np.log(4.0)

    simplex = np.array([[1.0, 1.0], [10.0, 1.0], [1.0, 10.0]])

    args = fit.nelder_mead(gradient, x, y, simplex)

    assert np.allclose(args, [3.0, 4.0], atol=1e-3)


def test_fit_normal():
    x = np.linspace(-10.0, 10.0, 100)
    y = normal.pdf(x, mu=2.3, sigma=3.4)
    args = fit.fit_normal(x, y)
    assert np.allclose(args, [2.3, 3.4, 1.0], atol=1e-3)

    y *= 0.5

    args = fit.fit_normal(x, y)
    assert np.allclose(args, [4.6, 6.8, 2.0], atol=1e-3)


def test_fit_lognormal():
    x = np.linspace(0.001, 30.0, 100)
    y = lognormal.pdf(x, mu=2.3, sigma=1.2)
    args = fit.fit_lognormal(x, y)
    assert np.allclose(args, [2.3, 1.2, 0.0], atol=1e-3)

    x += 10.0

    args = fit.fit_lognormal(x, y)
    assert np.allclose(args, [2.3, 1.2, 10.0], atol=1e-3)
