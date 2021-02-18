import numpy as np
# import matplotlib.pyplot as plt

from typing import Callable, List, Tuple


_s2 = np.sqrt(2.0)
_s2pi = np.sqrt(2.0 * np.pi)


def normal_pdf(x: np.ndarray, sigma: float, mu: float) -> np.ndarray:
    return 1.0 / (sigma * _s2pi) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def lognormal_pdf(x: np.ndarray, sigma: float, mu: float) -> np.ndarray:
    return 1.0 / (x * sigma * _s2pi) * np.exp(-0.5 * ((np.log(x) - mu) / sigma) ** 2)


def fit_pdf(
    x: np.ndarray,
    y: np.ndarray,
    pdf: Callable,
    args: List[np.ndarray],
) -> Tuple[Tuple[float, ...], float]:
    ndim = len(args) + 1
    dims = [tuple(i for i in range(ndim) if i != j) for j in range(len(args))]

    pdfs = pdf(x, *[np.expand_dims(arg, axis=dim) for arg, dim in zip(args, dims)])

    errs = np.sum(np.square(pdfs - y), axis=-1)
    minerr = np.argmin(errs)

    idx = np.unravel_index(minerr, errs.shape)
    opt = [arg[i] for arg, i in zip(args, idx)]

    return tuple(opt), errs[idx]


def fit_normal(x: np.ndarray, y: np.ndarray, n: int = 100):
    mean = x.mean()
    rng = (x.max() - x.min()) / 4.0
    mu = np.linspace(mean - rng, mean + rng, n)
    sigma = np.linspace(0.001, 10.0, n)

    return fit_pdf(x, y, normal_pdf, [sigma, mu])


def fit_lognormal(x: np.ndarray, y: np.ndarray, n: int = 100):
    mean = x.mean()
    rng = (x.max() - x.min()) / 4.0
    mu = np.linspace(mean - rng, mean + rng, n)
    sigma = np.linspace(0.001, 10.0, n)

    return fit_pdf(x, y, normal_pdf, [sigma, mu])


# sigma = np.random.random() * 10.0
# mu = np.random.random() * 10.0

# print("sigma", sigma, "mu", mu)
# dist = np.random.normal(mu, sigma, 1000)
# c, bins, _ = plt.hist(dist, bins=30, density=True)
# print(c)
# # hist = np.histogram(dist, bins=100, density=True)[0]
# # mean = np.mean(hist)

# # (fs, fm), e = fit_pdf(
# #     bins[:-1], c, normal_pdf, [np.linspace(0.01, 10.0, 100), np.linspace(0, 100, 100)]
# # )
# (fs, fm), e = fit_normal(bins[:-1], c)

# print("fit sigma", fs, "fit mu", fm)
# # pdf = normal_pdf(bins, 2.0, 5.0)


# # plt.plot(bins, pdf)
# # plt.axvline(mean)
# plt.show()
