from scipy.stats import poisson

from spcal.lib.spcalext import dists


def test_poisson_quantile():
    for q in [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]:
        for lam in [0.01, 1.0, 10.0, 100.0]:
            a = dists.poisson_quantile(q, lam)
            b = poisson.ppf(q, lam)
            assert a == b
