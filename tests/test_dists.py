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
    for lam in np.linspace(1.0, 10.0, 10):
        assert np.allclose(poisson.cdf(k, lam), stats.poisson.cdf(k, lam))
        assert np.allclose(poisson.pdf(k, lam), stats.poisson.pmf(k, lam))
