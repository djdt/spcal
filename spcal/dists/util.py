from importlib.resources import files

import numpy as np

from spcal.calc import interpolate_3d
from spcal.dists import lognormal, poisson

qtable = np.load(
    files("spcal.resources").joinpath("cpln_quantiles.npz").open("rb"),
    allow_pickle=False,
)


def zero_trunc_quantile(
    lam: np.ndarray | float, y: np.ndarray | float
) -> np.ndarray | float:
    k0 = np.exp(-lam)
    return np.maximum((y - k0) / (1.0 - k0), 0.0)


def compound_poisson_lognormal_quantile_lookup(
    q: float, lam: np.ndarray | float, sigma: float
) -> np.ndarray | float:
    """The quantile of a compound Poisson-Lognormal distribution.

    Interpolates values from a simulation of 1e10 zero-truncated values.
    The lookup table spans lambda values from 0.01 to 100.0, sigmas of
    0.3 to 0.6 and zt-quantiles of 1e-7 to 1.0 - 1e-7.

    Args:
        q: quantile
        lam: mean of the Poisson distribution
        sigma: log stddev of the log-normal distribution

    Returns:
        the ``q`` th value of the compound Poisson-Lognormal
    """
    lam = np.atleast_1d(lam)
    q0 = zero_trunc_quantile(lam, q)
    nonzero = q0 > 0.0

    qs = np.zeros_like(lam)
    qs[nonzero] = interpolate_3d(
        lam[nonzero],
        np.full_like(lam[nonzero], sigma),
        q0[nonzero],
        qtable["lambdas"],
        qtable["sigmas"],
        qtable["ys"],
        qtable["quantiles"],
    )
    if len(qs) == 1:
        qs = float(qs)
    return qs


def compound_poisson_lognormal_quantile_approximation(
    q: float, lam: float, mu: float, sigma: float
) -> float:
    """Appoximation of a compound Poisson-Lognormal quantile.

    Calculates the zero-truncated quantile of the distribution by appoximating the
    log-normal sum for each value ``k`` given by the Poisson distribution. The
    CDF is calculated for each log-normal, weighted by the Poisson PDF for ``k``.
    The quantile is taken from the sum of the CDFs.

    <5% error for lam < 100.0; sigma < 0.5

    Args:
        q: quantile
        lam: mean of the Poisson distribution
        mu: log mean of the log-normal distribution
        sigma: log stddev of the log-normal distribution

    Returns:
        the ``q`` th value of the compound Poisson-Lognormal
    """

    # A reasonable overestimate of the upper value
    uk = poisson.quantile(1.0 - 1e-12, lam)
    k = np.arange(0, uk + 1, dtype=int)
    pdf = poisson.pdf(k, lam)

    valid = np.isfinite(pdf)
    k = k[valid]
    pdf = pdf[valid]
    cdf = np.cumsum(pdf)

    # Calculate the zero-truncated quantile
    q0 = zero_trunc_quantile(lam, q)
    if q0 <= 0.0:  # The quantile is in the zero portion
        return 0.0
    weights = pdf[1:]
    k = k[1:]
    # Re-normalize weights
    weights /= weights.sum()

    # Get the sum LN for each value of the Poisson
    mus, sigmas = sum_iid_lognormals(k, np.log(1.0) - 0.5 * sigma**2, sigma)
    # The quantile of the last log-normal, must be lower than this
    upper_q = lognormal.quantile(q0, mus[-1], sigmas[-1])

    xs = np.linspace(lam, upper_q, 10000)
    cdf = np.sum(
        [w * lognormal.cdf(xs, m, s) for w, m, s in zip(weights, mus, sigmas)],
        axis=0,
    )
    q = xs[np.argmax(cdf > q0)]
    return q


def zero_truncated_poisson(lam: float, size: int) -> np.ndarray:
    """Poisson distribution with no zeros.

    Args:
        lam: lambda of non-zero truncated distribution
        size: size of output

    Returns:
        array of random values from the zero-truncated distribution
    """
    u = np.random.uniform(np.exp(-lam), size=size)
    return 1 + np.random.poisson(lam + np.log(u), size=size)


def simulate_zt_compound_poisson(
    lam: float, dist: np.ndarray, weights: np.ndarray | None = None, size: int = 100000
) -> np.ndarray:
    """Simulate a zero-truncated compound poisson distribution.

    The distribution is :math:`Y = \\sum_{n=1}^{N} X_n` where X is ``dist``
    and ``N`` is defined by a Poisson distribution with mean ``lam``.

    Args:
        lam: mean of the Poisson distribution
        dist: distribution to sample from
        weights: probabilities to draw sample from ``dist``
        size: size of simulation

    Returns:
        ``size`` points from Y
    """
    sim = np.zeros(size, dtype=np.float32)

    poi = zero_truncated_poisson(lam, size=size)
    unique, idx, counts = np.unique(poi, return_counts=True, return_inverse=True)
    for i, (u, c) in enumerate(zip(unique, counts)):
        sim[idx == i] += np.sum(np.random.choice(dist, size=(u, c), p=weights), axis=0)

    return sim


def sum_iid_lognormals(
    n: int | np.ndarray, mu: float, sigma: float, method: str = "Fenton-Wilkinson"
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Sum of ``n`` identical independant log-normal distributions.

    The sum is approximated by another log-normal distribution, defined by
    the returned parameters. By feaults, the Fenton-Wilkinson approximation is used
    for good right-tail accuracy [3]_.

    Args:
        n: int or array of ints
        mu: log mean of the underlying distributions
        sigma: log stddev of the underlying distributions
        method: approximation to use, 'Fenton-Wilkinson' or 'Lo'

    Returns:
        mu, sigma of the log-normal approximation

    References:
        .. [3] L. F. Fenton, "The sum of lognormal probability distributions in scatter
            transmission systems", IRE Trans. Commun. Syst., vol. CS-8, pp. 57-67, 1960.
            https://doi.org/10.1109/TCOM.1960.1097606
        .. C. F. Lo, "The sum and Difference of Two Lognormal Random Variables",
            Journal of Applied Mathematics, 2012, 838397.
            https://doi.org/10.1155/2012/838397
    """
    if method == "Fenton-Wilkinson":
        sigma2_x = np.log((np.exp(sigma**2) - 1.0) / n + 1.0)
        mu_x = np.log(n * np.exp(mu)) + 0.5 * (sigma**2 - sigma2_x)
        return mu_x, np.sqrt(sigma2_x)
    elif method == "Lo":
        Sp = n * np.exp(mu + 0.5 * sigma**2)
        sigma2_s = n / Sp**2 * sigma**2 * np.exp(mu + 0.5 * sigma**2) ** 2
        return np.log(Sp) - 0.5 * sigma2_s, np.sqrt(sigma2_s)
    else:
        raise NotImplementedError
