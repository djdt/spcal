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
        SPCalLimit.fromMethodString("Invalid", x, {}, {}, {})


def test_limit_from_poisson():
    lim = SPCalLimit.fromPoisson(x, alpha=0.001, max_iters=1)  # ld ~= 87
    assert lim.name == "Poisson"
    assert lim.params["alpha"] == 0.001
    limit = poisson.formula_c(np.mean(x), alpha=0.001)[0] + np.mean(x)
    if UPPER_INTEGER:
        limit = int(limit) + 1.0
    assert lim.detection_threshold == limit

    assert lim.accumulationLimit("signal mean") == lim.mean_signal
    assert lim.accumulationLimit("detection threshold") == lim.detection_threshold
    assert np.isclose(
        lim.accumulationLimit("half detection threshold"),
        (lim.mean_signal + lim.detection_threshold) / 2.0,
    )

    for name, func in zip(
        ["currie", "formula a", "formula c", "stapleton"],
        [
            poisson.currie,
            poisson.formula_a,
            poisson.formula_c,
            poisson.stapleton_approximation,
        ],
    ):
        lim = SPCalLimit.fromPoisson(x, alpha=0.001, max_iters=1, formula=name)
        limit = func(np.mean(x), alpha=0.001)[0] + np.mean(x)
        if UPPER_INTEGER:
            limit = int(limit) + 1.0
        assert lim.detection_threshold == limit


def test_limit_from_gaussian():
    lim = SPCalLimit.fromGaussian(x, alpha=0.001, max_iters=1)  # ld ~= 87
    assert lim.name == "Gaussian"
    assert lim.params["alpha"] == 0.001
    limit = np.mean(x) + np.std(x) * z
    # if UPPER_INTEGER:
    #     limit = int(limit) + 1.0
    assert np.isclose(lim.detection_threshold, limit)


def test_limit_from_compound_poisson_approximation():
    true_q = 75.5861212  # from simulated data (1e10 samples)
    sigma = 0.41

    lim = SPCalLimit.fromCompoundPoisson(
        x,
        alpha=0.001,
        method="approximation",
        max_iters=1,
        single_ion_dist=None,
        sigma=sigma,
    )
    assert lim.name == "CompoundPoisson"
    assert np.isclose(
        lim.detection_threshold, true_q, rtol=1e-2
    )  # from simulation of 1e9 samples


def test_limit_from_compound_poisson_lookup_table():
    true_q = 75.6605430  # from simulated data (1e9 samples)
    sigma = 0.405  # not in table

    lim = SPCalLimit.fromCompoundPoisson(
        x,
        alpha=0.001,
        method="lookup table",
        max_iters=1,
        single_ion_dist=None,
        sigma=sigma,
    )
    assert lim.name == "CompoundPoisson"
    assert np.isclose(
        lim.detection_threshold, true_q, rtol=1e-3
    )  # from simulation of 1e9 samples


def test_limit_from_compound_poisson_simulation():
    true_q = 75.5861212
    sigma = 0.41

    # Nu Instruments style SIS
    sis = np.random.lognormal(10.0, sigma, size=1000000)
    lim = SPCalLimit.fromCompoundPoisson(
        x,
        alpha=0.001,
        method="simulation",
        max_iters=1,
        single_ion_dist=sis,
    )
    assert np.isclose(
        lim.detection_threshold, true_q, rtol=1e-1
    )  # from simulation of 1e9 samples, lowered tolerance due to random error

    # TOFWERK style SIS
    hist, edges = np.histogram(sis, 512)
    sis = np.stack((edges[1:], hist), axis=1)

    lim = SPCalLimit.fromCompoundPoisson(
        x,
        alpha=0.001,
        method="simulation",
        max_iters=1,
        single_ion_dist=sis,
    )
    assert np.isclose(
        lim.detection_threshold, true_q, rtol=1e-1
    )  # from simulation of 1e9 samples, lowered tolerance due to random error


def test_limit_from_compound_poisson_errors():
    with pytest.raises(ValueError):
        SPCalLimit.fromCompoundPoisson(
            x,
            alpha=0.001,
            method="simulation",
            max_iters=1,
            single_ion_dist=None,
            sigma=0.4,
        )


def test_limit_windowed():
    lim = SPCalLimit.fromPoisson(x, window_size=3, max_iters=1)
    assert lim.params["window"] == 3
    assert lim.detection_threshold.size == x.size

    lim = SPCalLimit.fromGaussian(x, window_size=3, max_iters=1)
    assert lim.params["window"] == 3
    assert lim.detection_threshold.size == x.size


def test_limit_windowed_with_nan():
    y = x.astype(float)
    y[100:200] = np.nan

    lim = SPCalLimit.fromPoisson(y, window_size=3, max_iters=1)
    assert lim.params["window"] == 3
    assert lim.detection_threshold.size == y.size
    assert np.all(~np.isnan(lim.detection_limit[:100]))
    assert np.all(np.isnan(lim.detection_limit[100:200]))
    assert np.all(~np.isnan(lim.detection_limit[200:]))

    lim = SPCalLimit.fromGaussian(y, window_size=3, max_iters=1)
    assert lim.params["window"] == 3
    assert lim.detection_threshold.size == y.size
    assert np.all(~np.isnan(lim.detection_limit[:100]))
    assert np.all(np.isnan(lim.detection_limit[100:200]))
    assert np.all(~np.isnan(lim.detection_limit[200:]))


def test_limit_from():  # Better way for normality check?
    np.random.seed(987634)
    for lam in np.linspace(1.0, 100.0, 25):
        x = np.random.poisson(size=10000, lam=lam)
        lim_g = SPCalLimit.fromGaussian(x, max_iters=1)
        lim_p = SPCalLimit.fromPoisson(x, max_iters=1)
        lim_h = SPCalLimit.fromHighest(x, max_iters=1)
        lim_b = SPCalLimit.fromBest(x, max_iters=1)

        assert lim_h.name == max(lim_p, lim_g, key=lambda x: x.detection_threshold).name
        f = np.count_nonzero((x > 0.0) & (x <= 5.0)) / np.count_nonzero(x)
        assert lim_b.name == ("Poisson" if f > 0.05 else "Gaussian")

    # Make sure quad / tof detection works
    x = np.random.poisson(size=1000, lam=10.0)
    lim_c = SPCalLimit.fromBest(x, max_iters=1)
    assert lim_c.name == "Poisson"
    lim_c = SPCalLimit.fromBest(x / 10.0, max_iters=1)
    assert lim_c.name == "CompoundPoisson"


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


def test_limit_highest():
    x = np.random.poisson(size=1000, lam=1.0)
    lim = SPCalLimit.fromHighest(x, max_iters=1)
    assert lim.name == "Poisson"
    x = np.random.poisson(size=1000, lam=1000.0)
    lim = SPCalLimit.fromHighest(x, max_iters=1)
    assert lim.name == "Gaussian"


def test_limit_iterative():
    y = x.copy()
    idx = np.random.choice(1000, size=100, replace=False)
    y[idx] = y[idx] * 1000.0

    assert np.isclose(  # Estimates background within 1%
        SPCalLimit.fromPoisson(y, max_iters=10).mean_signal,
        50.0,
        atol=0.0,
        rtol=0.01,
    )

    assert np.isclose(  # Estimates background within 1%
        SPCalLimit.fromGaussian(y, alpha=0.001, max_iters=10).mean_signal,
        50.0,
        atol=0.0,
        rtol=0.01,
    )


def test_gaussian_error_rates():
    for mean in [20.0, 50.0, 100.0]:
        x = np.random.normal(mean, mean, size=10000)
        for alpha in [0.01, 0.05, 0.1]:
            limit = SPCalLimit.fromGaussian(x, alpha=alpha, max_iters=0)
            error_rate = np.count_nonzero(x > limit.detection_threshold) / x.size
            assert np.isclose(error_rate, alpha, atol=0.01)
