import numpy as np

from spcal.dists.normal import erf, erfinv


def cdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Cummulative density function of a log-normal distribution.

    Args:
        x: x values
        mu: mean of underlying normal distribution
        sigma: shape parameter

    Returns:
        CDF at all ``x``
    """
    return 0.5 * (1.0 + erf((np.log(x) - mu) / (sigma * np.sqrt(2.0))))


def pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Probabilty density function of a log-normal distribution.

    Args:
        x: x values
        mu: mean of underlying normal distribution
        sigma: shape parameter

    Returns:
        PDF at all ``x``
    """
    return (
        1.0
        / (x * sigma * np.sqrt(2.0 * np.pi))
        * np.exp(-0.5 * ((np.log(x) - mu) / sigma) ** 2)
    )


def moments(mu: float, sigma: float) -> tuple[float, float]:
    ex = np.exp(mu + 0.5 * sigma * sigma)
    vx = ex * ex * (np.exp(sigma * sigma) - 1.0)
    return ex, vx


def from_moments(ex: float, vx: float) -> tuple[float, float]:
    mu = np.log(ex / np.sqrt(1.0 + vx / (ex * ex)))
    sigma = np.sqrt(np.log(1.0 + vx / (ex * ex)))
    return mu, sigma


def quantile(quantile: float, mu: float, sigma: float) -> float:
    """Quantile (inverse CDF) function of a log-normal distribution.

    Args:
        quantile: values at which to evaluate
        mu: mean of underlying normal distribution
        sigma: shape parameter

    Returns:
        quantile at all ``quantile``
    """
    return np.exp(mu + np.sqrt(2.0 * sigma**2) * erfinv(2.0 * quantile - 1.0))
