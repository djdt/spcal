from statistics import NormalDist

import numpy as np

from spcal.dists import lognormal, normal, poisson


def compound_poisson_lognormal_quantile(
    q: float, lam: float, mu: float, sigma: float, method: str = "Fenton-Wilkinson"
) -> float:
    """Appoximation of a compound Poisson-Log-normal quantile.

    Calcultes the zero-truncated quantile of the distribution by appoximating the
    log-normal sum for each value ``k`` given by the Poisson distribution. The
    CDF is calculated for each log-normal, weighted by the Poisson PDF for ``k``.
    The quantile is taken from the sum of the CDFs.

    <1% error for lam < 50.0

    Args:
        q: quantile
        lam: mean of the Poisson distribution
        mu: log mean of the log-normal distribution
        sigma: log stddev of the log-normal distribution

    Returns:
        the ``q``th value of the compound Poisson-Log-normal
    """
    # A reasonable overestimate of the upper value
    uk = (
        int((lam + 1.0) * NormalDist().inv_cdf(1.0 - (1.0 - q) / 1e3) * np.sqrt(lam))
        + 1
    )
    k = np.arange(0, uk + 1)
    pdf = poisson.pdf(k, lam)
    cdf = np.cumsum(pdf)

    # Calculate the zero-truncated quantile
    q0 = (q - pdf[0]) / (1.0 - pdf[0])
    if q0 <= 0.0:  # The quantile is in the zero portion
        return 0.0
    # Trim values with a low probability
    valid = pdf > 1e-6
    weights = pdf[valid][1:]
    k = k[valid][1:]
    cdf = cdf[valid]
    # Re-normalize weights
    weights /= weights.sum()

    # Get the sum LN for each value of the Poisson
    mus, sigmas = [], []
    for _k in k:
        m, s = sum_iid_lognormals(
            _k, np.log(1.0) - 0.5 * sigma**2, sigma, method=method
        )
        mus.append(m)
        sigmas.append(s)

    mus, sigmas = np.asarray(mus), np.asarray(sigmas)
    # The quantile of the last log-normal, must be lower than this
    upper_q = lognormal.quantile(q, mus[-1], sigmas[-1])

    xs = np.linspace(1e-9, upper_q, 10000)
    cdf = np.sum(
        [w * lognormal.cdf(xs, m, s) for w, m, s in zip(weights, mus, sigmas)],
        axis=0,
    )
    q = xs[np.argmax(cdf > q0)]
    return q


def compound_poisson_lognormal_quantile_cdf(
    q: float, lam: float, mu: float, sigma: float, cdf_method: str = "Farley"
) -> float:
    """Appoximation of a compound Poisson-Log-normal quantile.

    Calcultes the zero-truncated quantile of the distribution by appoximating the
    log-normal sum for each value ``k`` given by the Poisson distribution. The
    CDF is calculated for each log-normal, weighted by the Poisson PDF for ``k``.
    The quantile is taken from the sum of the CDFs.

    <1% error for lam < 50.0

    Args:
        q: quantile
        lam: mean of the Poisson distribution
        mu: log mean of the log-normal distribution
        sigma: log stddev of the log-normal distribution

    Returns:
        the ``q``th value of the compound Poisson-Log-normal
    """
    # A reasonable overestimate of the upper value
    uk = (
        int((lam + 1.0) * NormalDist().inv_cdf(1.0 - (1.0 - q) / 1e3) * np.sqrt(lam))
        + 1
    )
    k = np.arange(0, uk + 1)
    pdf = poisson.pdf(k, lam)
    cdf = np.cumsum(pdf)

    # Calculate the zero-truncated quantile
    q0 = (q - pdf[0]) / (1.0 - pdf[0])
    if q0 <= 0.0:  # The quantile is in the zero portion
        return 0.0
    # Trim values with a low probability
    valid = pdf > 1e-6
    weights = pdf[valid][1:]
    k = k[valid][1:]
    cdf = cdf[valid]
    # Re-normalize weights
    weights /= weights.sum()

    # Get the sum LN for each value of the Poisson
    def farley_cdf(x: np.ndarray, mu: float, sigma: float, k: int) -> float:
        return 1.0 - (1.0 - normal.cdf((np.log(x) - mu) / sigma, 0.0, 1.0)) ** k

    if cdf_method == "Farley":
        cdf_func = farley_cdf
    else:
        raise NotImplementedError

    xs = np.linspace(1e-9, 10.0, 1000)
    cdf = np.sum([w * cdf_func(xs, mu, sigma, _k) for w, _k in zip(weights, k)], axis=0)
    q = xs[np.argmax(cdf > q0)]
    return q


def simulate_compound_poisson(
    lam: float, dist: np.ndarray, weights: np.ndarray | None = None, size: int = 100000
) -> np.ndarray:
    """Simulate a compound poisson distribution.

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

    poi = np.random.poisson(lam, size=size)
    unique, idx, counts = np.unique(poi, return_counts=True, return_inverse=True)
    for i, (u, c) in enumerate(zip(unique, counts)):
        sim[idx == i] += np.sum(np.random.choice(dist, size=(u, c), p=weights), axis=0)

    return sim


def sum_iid_lognormals(
    n: int | np.ndarray, mu: float, sigma: float, method: str = "Fenton-Wilkinson"
) -> tuple[np.ndarray, np.ndarray]:
    """Sum of ``n`` identical independant log-normal distributions.

    The sum is approximated by another log-normal distribution, defined by
    the returned parameters. Uses the Fenton-Wilkinson approximation for good
    right-tail accuracy.

    Args:
        n: int or array of ints
        mu: log mean of the underlying distributions
        sigma: log stddev of the underlying distributions

    Returns:
        mu, sigma of the log-normal approximation

    References:
        L. F. Fenton, "The sum of lognormal probability distributions in scatter
            transmission systems," IRE Trans. Commun. Syst., vol. CS-8, pp. 57-67, 1960.
            https://doi.org/10.1109/TCOM.1960.1097606
    """
    if method == "Fenton-Wilkinson":
        # Fenton-Wilkinson
        sigma2_x = np.log(
            (np.exp(sigma**2) - 1.0) * (n * np.exp(2.0 * mu)) / (n * np.exp(mu)) ** 2
            + 1.0
        )
        mu_x = np.log(n * np.exp(mu)) + 0.5 * (sigma**2 - sigma2_x)
        return mu_x, np.sqrt(sigma2_x)
    elif method == "Wu":
        aH = np.array(
            [
                0.27348104613815,
                0.82295144914466,
                1.38025853919888,
                1.95178799091625,
                2.54620215784748,
                3.17699916197996,
                3.86944790486012,
                4.68873893930582,
            ]
        )
        aH = np.stack((aH, -aH), axis=1).flat
        wH = np.array(
            [
                5.079294790166e-1,
                2.806474585285e-1,
                8.381004139899e-2,
                1.288031153551e-2,
                9.322840086242e-4,
                2.711860092538e-5,
                2.320980844865e-7,
                2.654807474011e-10,
            ]
        )
        wH = np.repeat(wH, 2)

        def psi(s: float, mu: float, sigma: float) -> float:
            return np.sum(
                wH
                / np.sqrt(np.pi)
                * np.exp(-s * np.exp(np.sqrt(2.0) * sigma * aH + mu))
            )

        s1, s2 = 0.001, 0.005

        def func(x, args):
            return [
                psi(s1, x[0], x[1]) - psi(s1, args[0], args[1]) ** n,
                psi(s2, x[0], x[1]) - psi(s2, args[0], args[1]) ** n,
            ]

        import warnings

        warnings.warn("warning: importing scipy")
        from scipy.optimize import fsolve

        res = fsolve(func, [np.log(n), 1.0], [mu, sigma])
        return res[0], res[1]

    elif method == "Lo":
        Sp = n * np.exp(mu + 0.5 * sigma**2)
        sigma2_s = n / Sp**2 * sigma**2 * np.exp(mu + 0.5 * sigma**2) ** 2
        return np.log(Sp) - 0.5 * sigma2_s, np.sqrt(sigma2_s)
    raise NotImplementedError
