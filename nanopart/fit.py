import numpy as np

from typing import Callable, List, Tuple


_s2 = np.sqrt(2.0)
_s2pi = np.sqrt(2.0 * np.pi)


def normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return 1.0 / (sigma * _s2pi) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# def lognormal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
#     return 1.0 / (x * sigma * _s2pi) * np.exp(-0.5 * ((np.log(x) - mu) / sigma) ** 2)


def fit_pdf(
    x: np.ndarray,
    y: np.ndarray,
    pdf: Callable,
    args: List[np.ndarray],
) -> Tuple[np.ndarray, float, Tuple[float, ...]]:
    ndim = len(args) + 1
    dims = [tuple(i for i in range(ndim) if i != j) for j in range(len(args))]

    pdfs = pdf(x, *[np.expand_dims(arg, axis=dim) for arg, dim in zip(args, dims)])

    errs = np.sum(np.square(pdfs - y), axis=-1)
    minerr = np.argmin(errs)

    idx = np.unravel_index(minerr, errs.shape)
    opt = [arg[i] for arg, i in zip(args, idx)]

    return pdfs[idx], errs[idx], tuple(opt)


def fit_normal(x: np.ndarray, y: np.ndarray, n: int = 100):
    mu = np.linspace(x.min(), x.max(), n)
    sigma = np.linspace(1e-3, 10.0, n)

    return fit_pdf(x, y, normal_pdf, [mu, sigma])


# def fit_lognormal(x: np.ndarray, y: np.ndarray, n: int = 100):
#     mu = np.linspace(0, np.log(x.max()), n)
#     print(mu)
#     sigma = np.linspace(1e-3, 10.0, n)
#     loc = np.linspace(x.min(), x.max(), n)

#     return fit_pdf(x, y, lognormal_pdf, [mu, sigma, loc])


# import matplotlib.pyplot as plt


# x = np.random.lognormal(0.1, 1.0, 1000) + 50.0

# c, bins, _ = plt.hist(x, bins=64, density=True)

# fit, err, opt = fit_lognormal(bins[1:], c)
# print(opt)

# plt.plot(bins[1:], fit)

# plt.show()
