from statistics import NormalDist

import numpy as np
import pytest

from spcal import poisson
from spcal.limit import (
    SPCalCompoundPoissonLimit,
    SPCalGaussianLimit,
    SPCalLimit,
    SPCalPoissonLimit,
)


@pytest.fixture(scope="module")
def poisson_data() -> np.ndarray:
    np.random.seed(690244)
    return np.random.poisson(lam=50.0, size=1000)


UPPER_INTEGER = True


def test_limit_errors():
    with pytest.raises(ValueError):
        SPCalLimit("test")
    with pytest.raises(NotImplementedError):
        SPCalLimit("test", signals=np.ones(10))


def test_limit_poisson(poisson_data: np.ndarray):
    mean = np.mean(poisson_data)

    lim = SPCalPoissonLimit(
        poisson_data, alpha=0.001, max_iterations=1, epsilon=0.5, function="currie"
    )  # ld ~= 87

    assert lim.name == "Poisson"
    assert lim.alpha == 0.001
    assert lim.eta == 2.0
    assert lim.epsilon == 0.5

    assert lim.parameters == {
        "function": "currie",
        "alpha": 0.001,
        "beta": 0.05,
        "eta": 2.0,
        "epsilon": 0.5,
    }

    limit = poisson.currie(mean, alpha=0.001, epsilon=0.5)[0] + mean  # type: ignore
    if UPPER_INTEGER:
        limit = np.ceil(limit)
    assert lim.mean_signal == mean
    assert lim.detection_threshold == limit

    for name, func in zip(
        ["formula a", "formula c", "stapleton"],
        [
            poisson.formula_a,
            poisson.formula_c,
            poisson.stapleton_approximation,
        ],
    ):
        lim = SPCalPoissonLimit(
            poisson_data, alpha=0.001, max_iterations=1, function=name
        )
        limit = func(mean, alpha=0.001)[0] + mean  # type: ignore
        if UPPER_INTEGER:
            limit = np.ceil(limit)
        assert lim.parameters["t_sample"] == 1.0
        assert lim.parameters["t_blank"] == 1.0
        assert lim.mean_signal == mean
        assert lim.detection_threshold == limit


def test_limit_gaussian(poisson_data: np.ndarray):
    z = NormalDist().inv_cdf(1.0 - 0.001)
    lim = SPCalGaussianLimit(poisson_data, alpha=0.001, max_iterations=1)  # ld ~= 87
    assert lim.name == "Gaussian"
    assert lim.parameters == {"alpha": 0.001}
    limit = np.mean(poisson_data) + np.std(poisson_data) * z
    assert np.isclose(lim.detection_threshold, limit)


def test_limit_gaussian_error_rates():
    np.random.seed(23468)
    x = np.random.normal(20.0, 5.0, size=100000)
    for alpha in [0.005, 0.01, 0.05, 0.1]:
        lim = SPCalGaussianLimit(x, alpha=alpha)
        num = np.count_nonzero(x > lim.detection_threshold)
        assert np.isclose(alpha, num / x.size, rtol=0.05)


def test_limit_is_gaussian_distributed():
    np.random.seed(928734)
    x = np.random.poisson(1.0, 1000)
    assert not SPCalGaussianLimit.isGaussianDistributed(x)
    x = np.random.poisson(5.0, 1000)
    assert not SPCalGaussianLimit.isGaussianDistributed(x)
    x = np.random.poisson(10.0, 1000)
    assert not SPCalGaussianLimit.isGaussianDistributed(x)
    x = np.random.poisson(15.0, 1000)
    assert SPCalGaussianLimit.isGaussianDistributed(x)
    x = np.random.poisson(100.0, 1000)
    assert SPCalGaussianLimit.isGaussianDistributed(x)


def test_limit_from_compound_poisson(poisson_data: np.ndarray):
    true_q = 75.5861212  # from simulated data (1e10 samples)
    sigma = 0.41

    lim = SPCalCompoundPoissonLimit(
        poisson_data,
        alpha=0.001,
        max_iterations=1,
        sigma=sigma,
    )
    assert lim.name == "Compound Poisson"
    assert lim.parameters == {"alpha": 0.001, "sigma": sigma}
    assert np.isclose(
        lim.detection_threshold, true_q, rtol=1e-2
    )  # from simulation of 1e9 samples

    true_q = 75.6605430  # from simulated data (1e9 samples)
    sigma = 0.405  # not in table

    lim = SPCalCompoundPoissonLimit(
        poisson_data,
        alpha=0.001,
        max_iterations=1,
        sigma=sigma,
    )
    assert np.isclose(
        lim.detection_threshold, true_q, rtol=1e-2
    )  # from simulation of 1e9 samples


def test_limit_windowed(poisson_data: np.ndarray):
    for limit_class in [
        SPCalPoissonLimit,
        SPCalGaussianLimit,
        SPCalCompoundPoissonLimit,
    ]:
        lim = limit_class(poisson_data, window_size=3, max_iterations=1)
        assert lim.window_size == 3
        assert isinstance(lim.detection_threshold, np.ndarray)
        assert lim.detection_threshold.size == poisson_data.size


def test_limit_windowed_with_nan(poisson_data: np.ndarray):
    data = poisson_data.astype(float)
    data[100:200] = np.nan

    for limit_class in [
        SPCalPoissonLimit,
        SPCalGaussianLimit,
        SPCalCompoundPoissonLimit,
    ]:
        lim = limit_class(data, window_size=3, max_iterations=1)
        assert isinstance(lim.detection_threshold, np.ndarray)
        assert lim.detection_threshold.size == data.size
        assert np.all(~np.isnan(lim.detection_threshold[:100]))
        assert np.all(np.isnan(lim.detection_threshold[100:200]))
        assert np.all(~np.isnan(lim.detection_threshold[200:]))


def test_limit_iterative(poisson_data: np.ndarray):
    y = poisson_data.astype(float)
    idx = np.random.choice(1000, size=100, replace=False)
    y[idx] = y[idx] * 1000.0

    for limit_class in [
        SPCalPoissonLimit,
        SPCalGaussianLimit,
        SPCalCompoundPoissonLimit,
    ]:
        lim = limit_class(y, max_iterations=10)

        assert np.isclose(  # Estimates background within 1%
            lim.mean_signal, 50.0, atol=0.0, rtol=0.01
        )


def test_limit_windowed_iterative(poisson_data: np.ndarray):
    y = poisson_data.astype(float)
    idx = np.random.choice(1000, size=100, replace=False)
    y[idx] = y[idx] * 1000.0

    for limit_class in [
        SPCalPoissonLimit,
        SPCalGaussianLimit,
        SPCalCompoundPoissonLimit,
    ]:
        lim = limit_class(y, max_iterations=10, window_size=10)

        assert lim.detection_threshold.size == 1000
