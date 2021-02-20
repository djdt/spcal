import numpy as np

from typing import Callable


_s2 = np.sqrt(2.0)
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

        if np.max(np.abs(fx[0] - fx[1:])) < ftol:
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


def fit_normal(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    def gradient(
        x: np.ndarray, y: np.ndarray, mu: float, sigma: float, scale: float
    ) -> float:
        return np.sum(np.abs(y - normal_pdf(x * scale, mu, sigma)))

    # Guess for result
    x0 = [np.mean(x), np.std(x), 0.0]
    simplex = np.ndarray(
        [x0, [np.max(x), x0[1], x0[2]], [x0[0], 10.0, x0[2]], [x0[0], x0[1], 10.0]]
    )

    fit = nelder_mead(gradient, x, y, simplex)
    return fit


import matplotlib.pyplot as plt


x = np.random.normal(10.0, 0.1, 1000)

c, bins, _ = plt.hist(x, bins=64, density=True)

sim = get_simplex([0.1, 0.1], [10, 10])

import time

s = time.time()
fit = nelder_mead(gradient_normal, bins[1:], c, sim)
print(time.time() - s)

print(fit)

# fit, err, opt = fit_lognormal(bins[1:], c)
# print(opt)

plt.plot(bins[1:], normal_pdf(bins[1:], *fit))

plt.show()
