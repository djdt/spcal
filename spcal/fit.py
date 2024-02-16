"""Distribution fitting calculations."""
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


def nelder_mead(
    func: Callable[..., float],
    x: np.ndarray,
    y: np.ndarray,
    simplex: np.ndarray,
    alpha: float = 1.0,
    gamma: float = 2.0,
    rho: float = 0.5,
    sigma: float = 0.5,
    max_iterations: int = 1000,
    stol: float = 1e-3,
    ftol: float = 1e-3,
) -> np.ndarray:
    """Function optimisation.

    Optimises parameters of ``func`` to fit ``func(x, params)`` to ``y``. The
    ``simplex`` is an array of best guesses and maximums for all params. The
    shape should be (n+1, n) for n params.

    Args:
        func: function to optimise
        x: x values (passed to func)
        y: `truth` values
        simplex: array of parameters
        alpha: reflection coefficient
        gamma: expansion coefficient
        rho: contraction coefficient
        sigma: shrink coefficient
        max_iterations: maximum fit attempts
        stol: minimum simplex difference
        ftol: minimum fit distance

    Return:
        optimal parameters for ``func``
    """
    # Calculate the values at each point in simplex
    fx = np.array([func(x, y, *s) for s in simplex])

    i = 1
    while i < max_iterations:
        # Reorder the simplex so that f(x0) < f(x1) < ... < f(xn+1)
        idx = np.argsort(fx)
        fx = fx[idx]
        simplex = simplex[idx]

        if (
            np.max(np.abs(simplex[0] - simplex[1:])) < stol
            and np.max(np.abs(fx[0] - fx[1:])) < ftol
        ):
            break

        # Centroid of all but n+1
        centroid = np.mean(simplex[:-1], axis=0)

        # Reflected point
        reflected = centroid + alpha * (centroid - simplex[-1])
        fxr = func(x, y, *reflected)

        if fx[0] <= fxr < fx[-2]:  # Reflection
            simplex[-1] = reflected
            fx[-1] = fxr
        elif fxr < fx[0]:  # Expansion
            expanded = centroid + gamma * (reflected - centroid)
            fxe = func(x, y, *expanded)
            if fxe < fxr:  # Jusk kidding, reflected
                simplex[-1] = expanded
                fx[-1] = fxe
            else:
                simplex[-1] = reflected
                fx[-1] = fxr
        else:  # Contraction
            contracted = centroid + rho * (simplex[-1] - centroid)
            fxc = func(x, y, *contracted)
            if fxc < fx[-1]:
                simplex[-1] = contracted
                fx[-1] = fxc
            else:  # Shrink
                simplex[1:] = simplex[0] + sigma * (simplex[1:] - simplex[0])
                fx[1:] = np.array([func(x, y, *s) for s in simplex[1:]])

        i += 1

    idx = np.argmax(fx)
    return simplex[idx]


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

    def gradient(
        x: np.ndarray, y: np.ndarray, mu: float, sigma: float, scale: float
    ) -> float:
        if mu <= 0 or sigma <= 0:
            return np.inf
        return np.sum(np.square(y - normal_pdf(x * scale, mu, sigma)))

    assert x.size == y.size
    # Guess for result
    mu = np.mean(x)
    s = np.std(x)
    simplex = np.array(
        [[mu, s, 1.0], [np.max(x), s, 1.0], [mu, 10.0, 1.0], [mu, s, 10.0]]
    )

    args = nelder_mead(gradient, x, y, simplex)
    return (args[0], args[1], args[2])


def fit_lognormal(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit a log-normal distribution to data.

    Args:
        x: x values
        y: y values

    Returns:
        optimal mu, sigma, loc
    """

    def gradient(
        x: np.ndarray, y: np.ndarray, mu: float, sigma: float, loc: float = 0.0
    ) -> float:
        xl = x + loc
        if sigma <= 0.0 or any(xl < 0.0):
            return np.inf
        return np.sum(np.square(y - lognormal_pdf(xl, mu, sigma)))

    assert x.size == y.size
    # Guess for result
    mu = np.log(np.median(x))
    s = 0.1
    simplex = np.array(
        [
            [mu, s, 0.0],
            [np.log(np.max(x)), s, 0.0],
            [mu, 1.0, 0.0],
            [mu, s, np.max(x) / 2.0],
        ]
    )

    args = nelder_mead(gradient, x, y, simplex)
    return (args[0], args[1], args[2])
