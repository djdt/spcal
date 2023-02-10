from statistics import NormalDist

import numpy as np

from spcal import poisson
from spcal.limit import SPCalLimit

np.random.seed(723423)
x = np.random.poisson(lam=50.0, size=1000)
z = NormalDist().inv_cdf(1.0 - 0.001)


# def test_limit_errors():
#     with pytest.raises(ValueError):
#         calc.calculate_limits(np.array([]), "Automatic")


def test_limit_from_poisson():
    lim = SPCalLimit.fromPoisson(x, alpha=0.001, max_iters=1)  # ld ~= 87
    assert lim.name == "Poisson"
    assert lim.params == {"alpha": 0.001}
    assert lim.detection_threshold == poisson.formula_c(np.mean(x), alpha=0.001)[
        0
    ] + np.mean(x)


def test_limit_from_gaussian():
    lim = SPCalLimit.fromGaussian(x, alpha=0.001, max_iters=1)  # ld ~= 87
    assert lim.name == "Gaussian"
    assert lim.params == {"alpha": 0.001}
    assert lim.detection_threshold == np.mean(x) + np.std(x) * z


def test_limit_from_highest():
    lim = SPCalLimit.fromHighest(x, max_iters=1)
    assert lim.name == "Poisson"


def test_limit_windowed():
    lim = SPCalLimit.fromPoisson(x, window_size=3, max_iters=1)
    assert lim.window_size == 3
    assert lim.detection_threshold.size == x.size


def test_limit_from():  # Better way for normality check?
    for lam in np.linspace(1.0, 100.0, 25):
        x = np.random.poisson(size=1000, lam=lam)
        lim_g = SPCalLimit.fromGaussian(x, max_iters=1)
        lim_p = SPCalLimit.fromPoisson(x, max_iters=1)
        lim_h = SPCalLimit.fromHighest(x, max_iters=1)
        lim_b = SPCalLimit.fromBest(x, max_iters=1)

        assert lim_h.name == max(lim_g, lim_p, key=lambda x: x.detection_threshold).name
        assert lim_b.name == ("Poisson" if lam < 50.0 else "Gaussian")


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
        SPCalLimit.fromGaussian(y, max_iters=10).mean_background,
        50.0,
        atol=0.0,
        rtol=0.01,
    )
