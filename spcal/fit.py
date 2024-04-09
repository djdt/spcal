"""Distribution fitting calculations."""
import warnings
from typing import Callable

import numpy as np


def normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Probabilty density function of a normal distribution."""
    return 1.0 / (sigma * np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def lognormal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Probabilty density function of a log-normal distribution."""
    return (
        1.0
        / (x * sigma * np.sqrt(2.0 * np.pi))
        * np.exp(-0.5 * ((np.log(x) - mu) / sigma) ** 2)
    )


def _central_finite_difference(
    fn, x: np.ndarray, eps: float | None = None
) -> np.ndarray:
    if eps is None:
        eps = np.sqrt(np.finfo(float).eps)

    g = np.empty_like(x)
    for i in range(x.size):
        x_f = x.copy()
        x_b = x.copy()
        x_f[i] += eps
        x_b[i] -= eps
        g[i] = (fn(x_f) - fn(x_b)) / (2.0 * eps)
    return g


def _line_search(
    fn,
    x: np.ndarray,
    g: np.ndarray,
    p: np.ndarray,
    a: float = 1.0,
    tau: float = 0.5,
    c1: float = 0.5,
    eps: float | None = None,
):
    if eps is None:
        eps = np.sqrt(np.finfo(float).eps)
    fx = fn(x)
    m = np.dot(g, p)
    while a > eps:
        if fx - fn(x + a * p) >= a * (-c1 * m):
            break
        a *= tau
    return a


def bfgs(
    fn: Callable[[np.ndarray], float],
    x0: np.ndarray,
    grad_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    eps: float = 1e-6,
    max_iter: int = 100,
):
    """Minimizes ``fn(x)`` using the Broyden-Fletcher-Goldfarb-Shanno algorithm.

    Gradients are estimated using the central finite difference.

    Args:
        fn: function to minimise
        x0: inital guess at function parameters
        eps: minimum function gradient
        max_iter: maximum number of iterations

    Returns:
        optimised values for x
    """
    if grad_fn is None:

        def grad_fn(x):
            return _central_finite_difference(fn, x)

    Id = np.identity(x0.size)
    Bk = np.linalg.inv(Id)
    iter = 0
    x = x0
    g = grad_fn(x)
    while iter < max_iter and np.linalg.norm(g) > eps:
        iter += 1

        p = -Bk @ g
        a = _line_search(fn, x, g, p)
        s = a * p

        gn = grad_fn(x + s)
        y = gn - g

        A = Id - ((s @ y.T) / (y.T @ s))
        B = Id - ((y @ s.T) / (y.T @ s))
        C = (s @ s.T) / (y.T @ s)
        Bk = A @ Bk @ B + C

        g = gn
        x = x + s
        print(f"iter {iter}: {x}, {g}")

    if iter == max_iter:
        warnings.warn("bfgs reached maximum iteration.")

    return x


# def get_simplex(mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
#     mins = np.array(mins)
#     simplex = np.repeat(mins[None, :], mins.size + 1, axis=0)
#     simplex[1:] += np.diag(maxs)
#     return simplex


def fit_normal(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit a normal distribution to data.

    Args:
        x: x values
        y: y values

    Returns:
        optimal mu, sigma, scale
    """

    def gradient(p: np.ndarray) -> float:
        mu, sigma = p[0], p[1]
        if sigma <= 0:
            return np.inf
        return np.sum(np.square(y - normal_pdf(x * p[2], mu, sigma)))

    assert x.size == y.size
    p0 = np.array([np.mean(x), np.std(x), 1.0])
    # Guess for result

    p = bfgs(gradient, p0)
    return p


def fit_lognormal(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit a log-normal distribution to data.

    Args:
        x: x values
        y: y values

    Returns:
        optimal mu, sigma, loc
    """

    def gradient(p: np.ndarray) -> float:
        if p[1] <= 0.0:
            return np.inf
        return np.sum(np.square(y - lognormal_pdf(x, p[0], p[1])))

    assert x.size == y.size
    # Guess for result
    p0 = np.array([np.log(np.median(x)), 0.5])

    p = bfgs(gradient, p0, eps=1e-3)
    print(p)
    return p
