from typing import Callable, Tuple

import numpy as np

# _s2 = np.sqrt(2.0)
_s2pi = np.sqrt(2.0 * np.pi)


def normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return 1.0 / (sigma * _s2pi) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def lognormal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return 1.0 / (x * sigma * _s2pi) * np.exp(-0.5 * ((np.log(x) - mu) / sigma) ** 2)


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

        if fx[0] <= fxr < fx[-2]:  # Relfection
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


def get_simplex(mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    mins = np.array(mins)
    simplex = np.repeat(mins[None, :], mins.size + 1, axis=0)
    simplex[1:] += np.diag(maxs)
    return simplex


def fit_normal(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    def gradient(
        x: np.ndarray, y: np.ndarray, mu: float, sigma: float, scale: float
    ) -> float:
        if mu <= 0 or sigma <= 0:
            return np.inf
        return np.sum(np.square(y - normal_pdf(x * scale, mu, sigma)))

    # Guess for result
    mu = np.mean(x)
    s = np.std(x)
    simplex = np.array(
        [[mu, s, 1.0], [np.max(x), s, 1.0], [mu, 10.0, 1.0], [mu, s, 10.0]]
    )

    args = nelder_mead(gradient, x, y, simplex)
    return normal_pdf(x * args[2], args[0], args[1]), gradient(x, y, *args), args


def fit_lognormal(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    def gradient(
        x: np.ndarray, y: np.ndarray, mu: float, sigma: float, loc: float = 0.0
    ) -> float:
        xl = x + loc
        if sigma <= 0.0 or any(xl < 0.0):
            return np.inf
        return np.sum(np.square(y - lognormal_pdf(xl, mu, sigma)))

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
    return lognormal_pdf(x + args[2], args[0], args[1]), gradient(x, y, *args), args
