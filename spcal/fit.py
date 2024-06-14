"""Distribution fitting calculations."""

import warnings
from typing import Callable

import numpy as np

from spcal.dists import lognormal, normal


def _central_finite_difference(
    fn, x: np.ndarray, eps: float | None = None
) -> np.ndarray:
    if eps is None:
        eps = np.sqrt(np.finfo(float).eps)

    g = np.empty_like(x)
    for i in range(x.size):
        h = max(eps * x[i], eps)
        x_f = x.copy()
        x_b = x.copy()
        x_f[i] += h
        x_b[i] -= h
        g[i] = (fn(x_f) - fn(x_b)) / (2.0 * h)
    return g


def _backtrack_line_search(
    fn,
    grad_fn,
    xk: np.ndarray,
    pk: np.ndarray,
    rho: float = 0.5,
    a: float = 1.0,
    c1: float = 1e-4,
    max_iter: int = 100,
) -> float:
    f0 = fn(xk)
    g0 = np.dot(grad_fn(xk), pk)

    iter = 1
    while iter < max_iter:
        if fn(xk + a * pk) <= f0 + c1 * a * g0:
            break
        a = rho * a
        iter += 1

    if iter == max_iter:
        warnings.warn("_backtrack_line_search reached maximum iteration")

    return a


def _zoom(
    fn,
    grad_fn,
    xk: np.ndarray,
    pk: np.ndarray,
    f0: np.ndarray,
    g0: np.ndarray,
    a_lo: float,
    a_hi: float,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_iter: int = 100,
):
    iter = 1

    while iter < max_iter:
        aj = (a_hi + a_lo) / 2.0  # bisect
        f_aj = fn(xk + aj * pk)
        if f_aj > f0 + c1 * aj * g0 or f_aj >= fn(xk + a_lo * pk):
            a_hi = aj
        else:
            g_aj = np.dot(grad_fn(xk + aj * pk), pk)
            if np.abs(g_aj) <= -c2 * g0:
                return aj
            if g_aj * (a_hi - a_lo) >= 0.0:
                a_hi = a_lo
            a_lo = aj
        iter += 1

    raise ValueError("_zoom max_iters exceeded")


def _line_search(
    fn,
    grad_fn,
    xk: np.ndarray,
    pk: np.ndarray,
    f0: np.ndarray | None = None,
    g0: np.ndarray | None = None,
    a: float = 1.0,
    amax: float = 100.0,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_iter: int = 100,
):
    if f0 is None:
        f0 = fn(xk)
    if g0 is None:
        g0 = np.dot(grad_fn(xk), pk)

    ap, ai = 0.0, a
    f_ap = 0.0

    iter = 1
    while iter < max_iter:
        f_ai = fn(xk + ai * pk)
        if np.all(f_ai > f0 + c1 * ai * g0) or (iter > 1 and f_ai >= f_ap):
            return _zoom(fn, grad_fn, xk, pk, f0, g0, ap, ai, c1=c1, c2=c2)
        f_ap = fn(xk + ap * pk)
        g_ai = np.dot(grad_fn(xk + ai * pk), pk)
        if np.all(np.abs(g_ai) <= c2 * np.abs(g_ai)):
            return ai
        if np.all(g_ai >= 0):
            return _zoom(fn, grad_fn, xk, pk, f0, g0, ai, ap, c1=c1, c2=c2)

        ap = ai
        ai = min(ai * 2.0, amax)
        iter += 1


def bfgs(
    fn: Callable[[np.ndarray], float],
    x0: np.ndarray,
    grad_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    alpha: float = 1.0,
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
            return _central_finite_difference(fn, x, eps=eps)

    Id = np.identity(x0.size)
    Hk = Id.copy()
    iter = 0
    x = x0
    gx = grad_fn(x)
    while iter < max_iter and np.linalg.norm(gx) > eps:
        p = -Hk @ gx
        # a = _backtrack_line_search(fn, grad_fn, x, p, a=alpha)
        a = _line_search(fn, grad_fn, x, p, amax=100.0, a=10.0)
        s = a * p

        gn = grad_fn(x + s)
        y = gn - gx

        rho = 1.0 / (y @ s.T)
        A = Id - (rho * s @ y.T)
        B = Id - (rho * y @ s.T)
        C = rho * s @ s.T
        Hk = A @ Hk @ B + C

        gx = gn
        x = x + s
        iter += 1
        print("iter", iter, x)

    if iter == max_iter:
        warnings.warn("bfgs reached maximum iteration")

    return x


def fit_normal(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit a normal distribution to data.

    Args:
        x: x values
        y: y values

    Returns:
        optimal mu, sigma, scale
    """

    def fn(p: np.ndarray) -> float:
        mu, sigma = p[0], p[1]
        if sigma <= 0:
            return np.inf
        return np.sum(np.square(y - normal.pdf(x * p[2], mu, sigma)))

    assert x.size == y.size
    x0 = np.array([np.mean(x), np.std(x), 1.0])
    # Guess for result

    p = bfgs(fn, x0, eps=1e-3)
    return p


def fit_lognormal(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit a log-normal distribution to data.

    Args:
        x: x values
        y: y values

    Returns:
        optimal mu, sigma, loc
    """

    def fn(p: np.ndarray) -> float:
        if p[1] <= 0.0:
            return np.inf
        return np.sum(np.square(y - lognormal.pdf(x + p[2], p[0], p[1])))

    assert x.size == y.size
    # Guess for result
    mu, sigma = lognormal.from_moments(np.mean(x), np.var(x))
    x0 = np.array([mu, sigma, 0.0])

    p = bfgs(fn, x0, eps=1e-3)
    return p
