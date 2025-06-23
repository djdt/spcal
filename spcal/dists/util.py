from importlib.resources import files

import numpy as np

from spcal.calc import expand_mask, interpolate_3d
from spcal.dists import lognormal, poisson

_qtable = np.load(
    files("spcal.resources").joinpath("cpln_quantiles.npz").open("rb"),
    allow_pickle=False,
)
# we must load the qtable data into memory to prevent errors on multithreaded read
qtable = {file: _qtable[file] for file in _qtable.files}


def zero_trunc_quantile(
    lam: np.ndarray | float, y: np.ndarray | float
) -> np.ndarray | float:
    """Returns the zero-truncated Poisson quantile.

    Args:
        lam: Poisson rate parameter(s)
        y: quantile(s) of non-truncated dist

    Returns:
        quantile(s) of the zero-truncated dist
    """
    k0 = np.exp(-lam)
    return np.maximum((y - k0) / (1.0 - k0), 0.0)


def compound_poisson_lognormal_quantile_lookup(
    q: np.ndarray | float,
    lam: np.ndarray | float,
    mu: np.ndarray | float,
    sigma: np.ndarray | float,
) -> np.ndarray | float:
    """The quantile of a compound Poisson-Lognormal distribution.

    Interpolates values from a simulation of 1e10 zero-truncated values.
    The lookup table spans lambda values from 0.01 to 100.0, sigmas of
    0.25 to 0.95 and zt-quantiles of 1e-3 to 1.0 - 1e-7.
    Maximum error is ~ 0.2 %.

    Args:
        q: quantile
        lam: mean of the Poisson distribution
        mu: log mean of the log-normal distribution
        sigma: log stddev of the log-normal distribution

    Returns:
        the ``q`` th value of the compound Poisson-Lognormal
    """
    lam = np.atleast_1d(lam)
    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)

    q0 = np.atleast_1d(zero_trunc_quantile(lam, q))
    nonzero = q0 > 0.0

    qs = np.zeros_like(lam)
    qs[nonzero] = interpolate_3d(
        lam[nonzero],
        sigma[nonzero],
        q0[nonzero],
        qtable["lambdas"],
        qtable["sigmas"],
        qtable["ys"],
        qtable["quantiles"],
    )
    # data collected for mean of 1.0 (mu = -0.5 * sigma**2) rescale to mu
    qs *= np.exp(mu + 0.5 * sigma**2)

    if len(qs) == 1:
        return qs[0]
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
    mus, sigmas = sum_iid_lognormals(k, mu, sigma)
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


def extract_compound_poisson_lognormal_parameters(
    x: np.ndarray, mask: np.ndarray | None = None
) -> np.ndarray:
    """Finds the parameters of compound-Poisson-lognormal distributed data, ``x``.

    .. math::
        N &\\sim Poisson(\\lambda) \\\\
        X &\\sim Lognormal(\\mu, \\sigma) \\\\
        Y &= \\sum{N}{n=1} X_{n}

    The value of :math:`\\lambda` is extracted using the percentage of zeros in ``x``.

    .. math::
        \\lambda = -\\log{P(0)}

    The expected value and variance of the underlying lognormal are extracted from the
    mean and variance of ``x``.

    .. math::
        E(Y) &= \\lambda E(X) \\\\
        V(Y) &= \\lambda E(X^2)

    Parameters :math:`\\mu` and :math:`\\sigma` are then extracted using the method of
    moments.

    Args:
        x: raw ICP-ToF signal of shape (samples, features)
        mask: mask of valid values, defaults to all non-nan

    Returns:
        array of [(lambda, mu, sigma), ...]
    """
    if mask is None:
        mask = ~np.isnan(x)
    else:
        mask = np.logical_and(~np.isnan(x), mask)

    zeros = np.count_nonzero(np.logical_and(mask, x == 0), axis=0)
    pzero = zeros / np.count_nonzero(mask, axis=0)
    lam = -np.log(pzero)

    if np.any(pzero > 1.0 - 1e-3):
        print("warning: only non zero points")
    elif np.any(pzero < 1e-3):
        print("warning: no zero values")

    mean = np.mean(x, axis=0, where=mask)

    EX = mean / lam
    EX2 = np.var(x, mean=mean, axis=0, where=mask) / lam

    mu = np.log(EX**2 / np.sqrt(EX2))
    sigma = np.sqrt(np.log(EX2 / EX**2))
    return np.asarray((lam, mu, sigma))


def extract_compound_poisson_lognormal_parameters_iterative(
    x: np.ndarray,
    alpha: float = 1e-4,
    dilation: int = 50,
    max_iters: int = 100,
    iter_eps: float = 1e-3,
    bounds: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Finds the parameters of compound Poisson -- lognormal distributed data, ``x``.

    Parameters are iterative found using ``extract_compound_poisson_lognormal_parameters``,
    a threshold based on these parameters is set then the parameters extracted again.
    This is repeated until either the threshold or both µ and σ no longer change.
    Parameters can be confined using the ``bounds`` argument, useful for reducing iterations
    in samples with many paraticles. By default only σ is bounded, 0.2 -- 1.0.

    Args:
        x: data
        alpha: alpha value to use during thresholding
        dilation: number of points to remove around detected peaks
        max_iters: maximum number of iterations
        iter_eps: smallest change in threshold allowed
        bounds: array of shape (3, 2) of parameter bounds

    Returns:
        lam, mu, sigma
    """

    threshold: float = np.inf
    previous_threshold: float = 0.0
    iters = 0

    params = np.full(3, np.inf)
    previous_params = np.zeros(3, dtype=float)

    if bounds is None:
        bounds = np.array([[0.0, np.inf], [-np.inf, np.inf], [0.2, 1.0]])

    while (
        (
            np.all(np.abs(previous_threshold - threshold) > iter_eps)
            and np.any(np.abs(previous_params[1:] - params[1:]) > iter_eps)
        )
        and iters < max_iters
    ) or iters == 0:
        previous_params = params
        previous_threshold = threshold

        mask = expand_mask(x > threshold, dilation)

        params = extract_compound_poisson_lognormal_parameters(x, ~mask)
        params = np.clip(params, bounds[:, 0], bounds[:, 1])

        threshold = compound_poisson_lognormal_quantile_lookup(  # type: ignore
            1.0 - alpha, params[0], params[1], params[2]
        )

        iters += 1

        if iters == max_iters and max_iters != 1:  # pragma: no cover
            print("iterative_extraction: reached max_iters")

    return params[0], params[1], params[2]
