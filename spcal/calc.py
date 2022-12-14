"""Misc and helper calculation functions."""
from bisect import bisect_left, insort
from typing import Dict, Tuple

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
