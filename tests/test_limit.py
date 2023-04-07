from statistics import NormalDist

import numpy as np
import pytest

from spcal import poisson
from spcal.limit import SPCalLimit

np.random.seed(690244)
x = np.random.poisson(lam=50.0, size=1000)
z = NormalDist().inv_cdf(1.0 - 0.001)


UPPER_INTEGER = True


def test_limit_errors():
    with pytest.raises(ValueError):
        SPCalLimit.fromMethodString("Invalid", x)


def test_limit_from_poisson():
    lim = SPCalLimit.fromPoisson(x, alpha=0.001, max_iters=1)  # ld ~= 87
    assert lim.name == "Poisson"
    assert lim.params == {"alpha": 0.001}
    limit = poisson.formula_c(np.mean(x), alpha=0.001)[0] + np.mean(x)
    if UPPER_INTEGER:
        limit = int(limit) + 1.0
    assert lim.detection_threshold == limit


def test_limit_from_gaussian():
    lim = SPCalLimit.fromGaussian(x, alpha=0.001, max_iters=1)  # ld ~= 87
    assert lim.name == "Gaussian"
    assert lim.params == {"alpha": 0.001}
    limit = np.mean(x) + np.std(x) * z
    if UPPER_INTEGER:
        limit = int(limit) + 1.0
    assert lim.detection_threshold == limit


def test_limit_windowed():
    lim = SPCalLimit.fromPoisson(x, window_size=3, max_iters=1)
    assert lim.window_size == 3
    assert lim.detection_threshold.size == x.size

    lim = SPCalLimit.fromGaussian(x, window_size=3, max_iters=1)
    assert lim.window_size == 3
    assert lim.detection_threshold.size == x.size


def test_limit_from():  # Better way for normality check?
    for lam in np.linspace(1.0, 100.0, 25):
        x = np.random.poisson(size=1000, lam=lam)
        lim_g = SPCalLimit.fromGaussian(x, max_iters=1)
        lim_p = SPCalLimit.fromPoisson(x, max_iters=1)
        lim_h = SPCalLimit.fromHighest(x, max_iters=1)
        lim_b = SPCalLimit.fromBest(x, max_iters=1)

        assert lim_h.name == max(lim_p, lim_g, key=lambda x: x.detection_threshold).name
        assert lim_b.name == ("Poisson" if lam < 10.0 else "Gaussian")


def test_limit_from_string():
    for string, method in zip(
        ["automatic", "highest", "gaussian", "poisson"],
        [
            SPCalLimit.fromBest,
            SPCalLimit.fromHighest,
            SPCalLimit.fromGaussian,
            SPCalLimit.fromPoisson,
        ],
    ):
        assert (
            SPCalLimit.fromMethodString(string, x, max_iters=1).detection_threshold
            == method(x, max_iters=1).detection_threshold
        )


def test_limit_iterative():
    y = x.copy()
    idx = np.random.choice(1000, size=100, replace=False)
    y[idx] = y[idx] * 1000.0

    assert np.isclose(  # Estimates background within 1%
        SPCalLimit.fromPoisson(y, max_iters=10).mean_background,
        50.0,
        atol=0.0,
        rtol=0.01,
    )

    assert np.isclose(  # Estimates background within 1%
        SPCalLimit.fromGaussian(y, alpha=0.001, max_iters=10).mean_background,
        50.0,
        atol=0.0,
        rtol=0.01,
    )
