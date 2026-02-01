from pathlib import Path

import numpy as np
from scipy import stats
from scipy.special import erf as erf_sp
from scipy.special import erfinv as erfinv_sp

from spcal.dists import lognormal, normal, poisson, util


# Test approximations are within their defined maximum errors
def test_erf():
    x = np.linspace(-10.0, 10.0, 1000)
    assert np.allclose(normal.erf(x), erf_sp(x), atol=1.5e-7)
    assert np.isclose(normal.erf(0.0), erf_sp(0.0), atol=1.5e-7)


def test_erfinv():
    x = np.linspace(-1.0, 1.0, 1000)
    assert np.allclose(normal.erfinv(x), erfinv_sp(x), atol=1.5e-9 / np.sqrt(2))
    assert np.isclose(normal.erfinv(0.5), erfinv_sp(0.5), atol=1.5e-9 / np.sqrt(2))


def test_standard_normal():
    q = np.linspace(0.0, 1.0, 1000)
    assert np.allclose(
        normal.standard_quantile(q), stats.norm.ppf(q, loc=0.0, scale=1.0), atol=1.5e-9
    )


def test_dist_normal():
    x = np.linspace(-100, 100, 3)
    mu = 5.0
    sigma = 2.0

    assert np.allclose(
        normal.cdf(x, mu, sigma), stats.norm.cdf(x, loc=mu, scale=sigma), atol=1e-3
    )
    assert np.allclose(normal.pdf(x, mu, sigma), stats.norm.pdf(x, loc=mu, scale=sigma))
    q = np.linspace(0.1, 0.9, 10)
    assert np.allclose(
        normal.quantile(q, mu, sigma), stats.norm.ppf(q, loc=mu, scale=sigma)
    )


def test_dist_lognormal():
    x = np.linspace(1e-6, 100, 1000)
    mu = np.log(5.0)
    sigma = 2.0

    assert np.allclose(
        lognormal.cdf(x, mu, sigma),
        stats.lognorm.cdf(x, sigma, scale=np.exp(mu)),
        atol=1.5e-9,
    )
    assert np.allclose(
        lognormal.pdf(x, mu, sigma), stats.lognorm.pdf(x, sigma, scale=np.exp(mu))
    )

    # central moments
    sx, vx = stats.lognorm.stats(sigma, scale=np.exp(mu), moments="mv")
    assert np.allclose(lognormal.moments(mu, sigma), (sx, vx))

    assert lognormal.from_moments(*lognormal.moments(mu, sigma)) == (mu, sigma)

    q = np.linspace(1e-6, 0.9, 10)
    assert np.allclose(
        lognormal.quantile(q, mu, sigma),
        stats.lognorm.ppf(q, sigma, scale=np.exp(mu)),
    )


def test_dist_poisson():
    k = np.arange(0, 100, dtype=int)
    for lam in np.linspace(0.1, 10.0, 100):
        assert np.allclose(poisson.cdf(k, lam), stats.poisson.cdf(k, lam))
        assert np.allclose(poisson.pdf(k, lam), stats.poisson.pmf(k, lam))

    k = np.arange(10, 20, dtype=int)
    assert np.allclose(poisson.cdf(k, 1.0), stats.poisson.cdf(k, 1.0))
    assert np.allclose(poisson.pdf(k, 1.0), stats.poisson.pmf(k, 1.0))

    qs = np.arange(0.1, 1.0, 0.01)
    for q in qs:
        assert np.isclose(poisson.quantile(q, lam), stats.poisson.ppf(q, lam))  # type: ignore


def test_compound_poisson_lognormal_quantile_approximation():
    lambdas = [0.01, 0.1, 1.0, 10.0, 100.0]
    sigmas = [0.3, 0.4, 0.5]
    qs = 1.0 - np.array([1e-3, 1e-4, 1e-5, 1e-6])

    for lam in lambdas:
        for sigma in sigmas:
            for q in qs:
                mu = np.log(1.0) - 0.5 * sigma**2
                qtrue = util.compound_poisson_lognormal_quantile_lookup(
                    q, lam, mu, sigma
                )
                qaprx = util.compound_poisson_lognormal_quantile_approximation(
                    q, lam, mu, sigma
                )
                # within 5 %
                assert np.isclose(qaprx, qtrue, rtol=0.05)


def test_compound_poisson_lognormal_quantile_approximation_zero_trunc():
    q = util.compound_poisson_lognormal_quantile_approximation(0.1, 0.001, 1.0, 0.5)
    assert q == 0


def test_extract_compound_poisson_lognormal_parameters():
    data = np.load(Path(__file__).parent.joinpath("data/cpln_simulations.npz"))
    for file in data:
        params = [float(x) for x in file.split(",")]
        predicted = util.extract_compound_poisson_lognormal_parameters(data[file])
        assert np.allclose(params, predicted, atol=0.01)


def test_extract_compound_poisson_lognormal_parameters_iterative():
    data = np.load(Path(__file__).parent.joinpath("data/cpln_simulations.npz"))
    for file in data:
        params = [float(x) for x in file.split(",")]
        x = data[file]

        predicted = util.extract_compound_poisson_lognormal_parameters_iterative(
            x, alpha=1e-4, iter_eps=1e-3
        )
        assert np.allclose(params, predicted, atol=0.02)

    data = np.load(Path(__file__).parent.joinpath("data/cpln_simulations_peaks.npz"))
    for file in data:
        params = [float(x) for x in file.split(",")]
        x = data[file]

        predicted = util.extract_compound_poisson_lognormal_parameters_iterative(
            x, alpha=1e-4, iter_eps=1e-3
        )
        assert np.allclose(params, predicted, atol=0.02)
