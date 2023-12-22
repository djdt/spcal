import numpy as np
from spcal.calc import erf, erfinv


def cdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return 0.5 * (1.0 + erf((np.log(x) - mu) / (sigma * np.sqrt(2.0))))


def pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Probabilty density function of a log-normal distribution."""
    return (
        1.0
        / (x * sigma * np.sqrt(2.0 * np.pi))
        * np.exp(-0.5 * ((np.log(x) - mu) / sigma) ** 2)
    )


def shifted_pdf(
    x: np.ndarray, mu: float, sigma: float, theta: float = 0.0
) -> np.ndarray:
    return pdf(x - theta, mu, sigma)


def moments(mu: float, sigma: float) -> tuple[float, float]:
    ex = np.exp(mu + 0.5 * sigma * sigma)
    vx = ex * ex * (np.exp(sigma * sigma) - 1.0)
    return ex, vx


def from_moments(ex: float, vx: float) -> tuple[float, float]:
    mu = np.log(ex / np.sqrt(1.0 + vx / (ex * ex)))
    sigma = np.sqrt(np.log(1.0 + vx / (ex * ex)))
    return mu, sigma


def quantile(quantile: float, mu: float, sigma: float) -> float:
    return np.exp(mu + np.sqrt(2.0 * sigma**2) * erfinv(2.0 * quantile - 1.0))
