import numpy as np

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

    return fit_pdf(x, y, normal_pdf, [sigma, mu])


def fit_lognormal(x: np.ndarray, y: np.ndarray, n: int = 100):
    mu = np.log(np.linspace(x.min(), x.max(), n))
    sigma = np.linspace(1e-3, 1.0, n)

    return fit_pdf(x, y, lognormal_pdf, [sigma, mu])
