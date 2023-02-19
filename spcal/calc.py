"""Misc and helper calculation functions."""
from bisect import bisect_left, insort
from typing import Tuple

import numpy as np

try:
    import bottleneck as bn

    BOTTLENECK_FOUND = True
except ImportError:  # pragma: no cover
    BOTTLENECK_FOUND = False


def moving_mean(x: np.ndarray, n: int) -> np.ndarray:
    """Calculates the rolling mean.

    Uses bottleneck.move_mean if available otherwise np.cumsum based algorithm.

    Args:
        x: array
        n: window size
    """
    if BOTTLENECK_FOUND:  # pragma: no cover
        return bn.move_mean(x, n)[n - 1 :]
    r = np.cumsum(x)
    r[n:] = r[n:] - r[:-n]  # type: ignore
    return r[n - 1 :] / n


def moving_median(x: np.ndarray, n: int) -> np.ndarray:
    """Calculates the rolling median.

    Uses bottleneck.move_median if available otherwise sort based algorithm.

    Args:
        x: array
        n: window size
    """
    if BOTTLENECK_FOUND:  # pragma: no cover
        return bn.move_median(x, n)[n - 1 :]

    r = np.empty(x.size - n + 1, x.dtype)
    sort = sorted(x[:n])
    m = n // 2
    m2 = m + n % 2 - 1

    for start in range(x.size - n):
        r[start] = sort[m] + sort[m2]
        end = start + n
        del sort[bisect_left(sort, x[start])]
        insort(sort, x[end])

    r[-1] = sort[m] + sort[m2]
    return r / 2.0


def moving_std(x: np.ndarray, n: int) -> np.ndarray:
    """Calculates the rolling standard deviation.

    Uses bottleneck.move_std if available otherwise np.cumsum based algorithm.

    Args:
        x: array
        n: window size
    """
    if BOTTLENECK_FOUND:  # pragma: no cover
        return bn.move_std(x, n)[n - 1 :]

    sums = np.empty(x.size - n + 1)
    sqrs = np.empty(x.size - n + 1)

    tab = np.cumsum(x) / n
    sums[0] = tab[n - 1]
    sums[1:] = tab[n:] - tab[:-n]

    tab = np.cumsum(x * x) / n
    sqrs[0] = tab[n - 1]
    sqrs[1:] = tab[n:] - tab[:-n]

    return np.sqrt(sqrs - sums * sums)


def otsu(x: np.ndarray, remove_nan: bool = False, nbins: str | int = "fd") -> float:
    """Calculates the otsu threshold.

    The Otsu threshold minimises intra-class variance for a two class system.
    If `remove_nan` then all nans are removed before computation.

    Args:
        x: array
        remove_nan: remove nan values
        nbins: number of bins to use

    See Also:
        :func:`skimage.filters.threshold_otsu`
    """
    if remove_nan:
        x = x[~np.isnan(x)]

    hist, bin_edges = np.histogram(x, bins=nbins)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0

    w1 = np.cumsum(hist)
    w2 = np.cumsum(hist[::-1])[::-1]

    u1 = np.cumsum(hist * bin_centers) / w1
    u2 = (np.cumsum((hist * bin_centers)[::-1]) / w2[::-1])[::-1]

    i = np.argmax(w1[:-1] * w2[1:] * (u1[:-1] - u2[1:]) ** 2)
    return bin_centers[i]


def pca(
    x: np.ndarray, trim_to_components: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    def standardise(x: np.ndarray) -> np.ndarray:
        return (x - x.mean(axis=0)) / x.std(axis=0)

    x = standardise(x)
    # evalues, evectors = np.linalg.eig(cov)
    u, s, v = np.linalg.svd(x, full_matrices=False)

    # ensure determenistic, see scikit-learn's svd_flip
    sign = np.sign(u[np.argmax(np.abs(u), axis=0), np.arange(u.shape[1])])
    u *= sign
    v *= sign[:, None]

    explained_variance = s**2 / x.shape[0]
    explained_variance = explained_variance / np.sum(explained_variance)

    return (
        u[:trim_to_components] * s[:trim_to_components],
        v[:trim_to_components],
        explained_variance[:trim_to_components],
    )
