from statistics import NormalDist

import numpy as np

from spcal import poisson
from spcal.limit import SPCalLimit

x = np.random.poisson(lam=50.0, size=1000)
z = NormalDist().inv_cdf(1.0 - 0.001)


# def test_limit_errors():
#     with pytest.raises(ValueError):
#         calc.calculate_limits(np.array([]), "Automatic")


def test_limit_from_poisson():
    lim = SPCalLimit.fromPoisson(x, alpha=0.001)  # ld ~= 87
    assert lim.name == "Poisson"
    assert lim.params == {"alpha": 0.001}
    assert lim.detection_threshold == poisson.formula_c(np.mean(x), alpha=0.001)[
        0
    ] + np.mean(x)


def test_limit_from_gaussian():
    lim = SPCalLimit.fromGaussian(x, alpha=0.001)  # ld ~= 87
    assert lim.name == "Gaussian"
    assert lim.params == {"alpha": 0.001}
    assert lim.detection_threshold == np.mean(x) + np.std(x) * z

    lim = SPCalLimit.fromGaussian(x, alpha=0.001, use_median=True)  # ld ~= 87
    assert lim.name == "Gaussian Median"
    assert lim.params == {"alpha": 0.001}
    assert lim.detection_threshold == np.median(x) + np.std(x) * z


def test_limit_from_highest():
    lim = SPCalLimit.fromHighest(x, alpha=0.001)
    assert lim.name == "Poisson"


def test_limit_windowed():
    lim = SPCalLimit.fromPoisson(x, window_size=3)
    assert lim.window_size == 3
    assert lim.detection_threshold.size == x.size


def test_limit_from_best():  # Better way for normality check?
    for lam in np.linspace(1.0, 100.0, 25):
        x = np.random.poisson(size=1000, lam=lam)
        lim = SPCalLimit.fromBest(x, alpha=0.001)
        assert lim.name == ("Poisson" if lam < 50.0 else "Gaussian")
