import numpy as np
from scipy import stats

from spcal.dists import lognormal, poisson, util


def test_dist_lognormal():
    x = np.linspace(0.001, 100, 100)
    mu = np.log(5.0)
    sigma = 2.0

    assert np.allclose(  # low accuracy at very low cdf values due to erf implementation
        lognormal.cdf(x, mu, sigma),
        stats.lognorm.cdf(x, sigma, scale=np.exp(mu)),
        atol=0.001,
    )
    assert np.allclose(
        lognormal.pdf(x, mu, sigma), stats.lognorm.pdf(x, sigma, scale=np.exp(mu))
    )
