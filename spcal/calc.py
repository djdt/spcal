"""Misc and helper calculation functions."""
from typing import Tuple

import numpy as np


def erf(x: float | np.ndarray) -> float | np.ndarray:
    """Error function approximation.

    The maximum error is 5e-4 [1].

    Args:
        x: value

    Returns:
        approximation of error function

    References:
        .. [1] Abramowitz, Milton, and Irene A. Stegun, eds. Handbook of mathematical
            functions with formulas, graphs, and mathematical tables.
            Vol. 55. US Government printing office, 1970.
    """
    assert np.all(x >= 0.0)
    # Maximum error: 2.5e-5
    a = np.array([0.278393, 0.230389, 0.000972, 0.078108])
    p = np.array([[1, 2, 3, 4]]).T
    sum = np.sum(a * np.power(x, p).T, axis=1)
    return 1.0 - 1.0 / (1.0 + sum) ** 4


def erfinv(x: float | np.ndarray) -> float | np.ndarray:
    """Inverse error function approximation.

    The maximum error is 6e-3 [2].

    Args:
        x: value

    Returns:
        approximation of inverse error function

    References:
        .. [2] Winitzki, S. A handy approximation for the error function and its inverse
            2008
    """
    sign = np.sign(x)
    x = np.log((1.0 - x) * (1.0 + x))

    tt1 = 2.0 / (np.pi * 0.147) + 0.5 * x
    tt2 = 1.0 / 0.147 * x

    return sign * np.sqrt(-tt1 + np.sqrt(tt1 * tt1 - tt2))


def gamma(x: float) -> float:
    """Gamma function approximation.

    Maximum error of 3e-7 [3].

    Args:
        x: value

    Returns:
        approximation of error function

    References:
        .. [3] Abramowitz, Milton, and Irene A. Stegun, eds. Handbook of mathematical
            functions with formulas, graphs, and mathematical tables.
            Vol. 55. US Government printing office, 1970.
    """
    assert x >= 0.0
    # Use recursion
    b = np.array(
        [
            1.0,
            -0.577191652,
            0.988205891,
            -0.897056937,
            0.918206857,
            -0.756704078,
            0.482199394,
            -0.193527818,
            0.035868343,
        ]
    )
    z = x % 1.0
    n = 1.0 / x if x < 1.0 else np.prod(z + np.arange(1, int(x - z)))
    return n * np.sum(b * np.power(z, np.arange(9)))


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
    """Perform a PCA on 'x', standard scales data.

    Args:
        x: input array of shape (samples, features)
        trim_to_components: trim to this many dims

    Returns:
        pca data points
        component vectors
        explained variance per dim
    """

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
        u[:, :trim_to_components] * s[:trim_to_components],
        v[:trim_to_components],
        explained_variance[:trim_to_components],
    )


def weights_from_weighting(
    x: np.ndarray, weighting: str, safe: bool = True
) -> np.ndarray:
    """Get weighting for `x`.

    Conveience function for calculating simple weightings. If `safe` then any
    zeros in `x` are replace with the minimum non-zero value.

    Args:
        x: 1d-array
        weighting: weighting string {'Equal', 'x', '1/x', '1/(x^2)'}
        safe: replace zeros with minimum

    Returns:
        weights, same size as x
    """
    if x.size == 0:
        return np.empty(0, dtype=x.dtype)

    if safe:
        if np.all(x == 0):  # Impossible weighting
            return np.ones_like(x)
        x = x.copy()
        x[x == 0] = np.nanmin(x[x != 0])

    if weighting == "Equal":
        return np.ones_like(x)
    elif weighting == "x":
        return x
    elif weighting == "1/x":
        return 1.0 / x
    elif weighting == "1/(x^2)":
        return 1.0 / (x**2.0)
    else:
        raise ValueError(f"Unknown weighting {weighting}.")


def weighted_rsq(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    """Calculate r² for weighted linear regression.

    Args:
        x: 1d-array
        y: array, same size as `x`
        w: weights, same size as `x`
    """
    c = np.cov(x, y, aweights=w)
    d = np.diag(c)
    stddev = np.sqrt(d.real)
    c /= stddev[:, None]
    c /= stddev[None, :]

    np.clip(c.real, -1, 1, out=c.real)
    if np.iscomplexobj(c):  # pragma: no cover
        np.clip(c.imag, -1, 1, out=c.imag)

    return c[0, 1] ** 2.0


def weighted_linreg(
    x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None
) -> Tuple[float, float, float, float]:
    """Weighted linear regression.

    Uses polyfit with sqrt(weights) for intercept and gradient.

    Args:
        x: 1d-array
        y: array, same size as `x`
        w: weights, same size as `x`

    Returns:
       gradient
       intercept
       r²
       error, S(y,x) the (unweighted) residual standard deviation

    See Also:
        :func:`pewlib.calibration.weighted_rsq`
    """
    coef = np.polynomial.polynomial.polyfit(x, y, 1, w=w if w is None else np.sqrt(w))
    r2 = weighted_rsq(x, y, w)
    if x.size > 2:
        error = np.sqrt(np.sum(((coef[0] + x * coef[1]) - y) ** 2) / (x.size - 2))
    else:
        error = 0.0

    return coef[1], coef[0], r2, error
