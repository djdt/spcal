"""Misc and helper calculation functions."""

import numpy as np


def is_integer_or_near(x: np.ndarray, max_deviation: float = 1e-3) -> np.ndarray:
    """Test if float data is 'near' integer.
    Near integers values are those less than `max_deviation` from a whole number.

    Args:
        x: float array
        max_deviation: max distance from whole number

    Returns:
        array of bool
    """
    if max_deviation < 0.0 or max_deviation >= 1.0:
        raise ValueError("'max_deviation' must be in the range [0-1).")
    return np.abs(x - np.round(x)) <= max_deviation


def search_sorted_closest(
    x: np.ndarray, v: np.ndarray, check_max_diff: float | None = None
):
    """Get the idx of the closest values in ``x`` for ``v``.

    If ``check_max_diff`` is a value, the maximum distance must be lower.

    Args:
        x: sorted array
        y: values to find closest idx of in x
        check_max_diff: if not None, check maximum diff is less than this

    Returns:
        idx of closest ``x`` values for ``v``

    Raises:
        ValueError if ``check_max_diff`` is not None and max diff greater.
    """
    idx = np.searchsorted(x, v, side="left")
    prev_less = np.abs(v - x[np.maximum(idx - 1, 0)]) < np.abs(
        v - x[np.minimum(idx, len(x) - 1)]
    )
    prev_less = (idx == len(x)) | prev_less
    idx[prev_less] -= 1

    if check_max_diff is not None:
        diffs = np.abs(x[idx] - v)
        if np.any(diffs > check_max_diff):
            raise ValueError("could not find value closer than 'check_max_diff'")

    return idx


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
    if remove_nan:  # pragma: no cover
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    stddev[stddev == 0.0] = np.nan

    c /= stddev[:, None]
    c /= stddev[None, :]

    np.clip(c.real, -1, 1, out=c.real)
    if np.iscomplexobj(c):  # pragma: no cover
        np.clip(c.imag, -1, 1, out=c.imag)

    return c[0, 1] ** 2.0


def weighted_linreg(
    x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None
) -> tuple[float, float, float, float]:
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
    coef, _ = np.polynomial.polynomial.polyfit(
        x, y, 1, w=w if w is None else np.sqrt(w), full=True
    )
    r2 = weighted_rsq(x, y, w)
    if x.size > 2:
        error = np.sqrt(np.sum(((coef[0] + x * coef[1]) - y) ** 2) / (x.size - 2))
    else:
        error = 0.0

    return coef[1], coef[0], r2, error


def interpolate_3d(
    x: np.ndarray | float,
    y: np.ndarray | float,
    z: np.ndarray | float,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    data: np.ndarray,
) -> np.ndarray | float:
    """Cubic interpolation of (x, y, z) for data.

    Args:
        x: x position of values to interpolate
        y: y position of values to interpolate
        z: z position of values to interpolate
        xs: x values of ``data``
        ys: y values of ``data``
        zs: z values of ``data``
        data: known values, shape (xs, ys, zs)

    Returns:
        interpolated values, shape (x)
    """
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    assert x.size == y.size == z.size
    assert data.shape == (xs.size, ys.size, zs.size)

    idx0 = np.searchsorted(xs, x, side="right") - 1
    idy0 = np.searchsorted(ys, y, side="right") - 1
    idz0 = np.searchsorted(zs, z, side="right") - 1

    idx1 = np.minimum(idx0 + 1, xs.size - 1)
    idy1 = np.minimum(idy0 + 1, ys.size - 1)
    idz1 = np.minimum(idz0 + 1, zs.size - 1)

    # Fix any edge cases
    idx0 = np.where(idx1 == xs.size - 1, idx1 - 1, idx0)
    idy0 = np.where(idy1 == ys.size - 1, idy1 - 1, idy0)
    idz0 = np.where(idz1 == zs.size - 1, idz1 - 1, idz0)

    xd = (x - xs[idx0]) / (xs[idx1] - xs[idx0])
    yd = (y - ys[idy0]) / (ys[idy1] - ys[idy0])
    zd = (z - zs[idz0]) / (zs[idz1] - zs[idz0])

    c00 = data[idx0, idy0, idz0] * (1.0 - xd) + data[idx1, idy0, idz0] * xd
    c01 = data[idx0, idy0, idz1] * (1.0 - xd) + data[idx1, idy0, idz1] * xd
    c10 = data[idx0, idy1, idz0] * (1.0 - xd) + data[idx1, idy1, idz0] * xd
    c11 = data[idx0, idy1, idz1] * (1.0 - xd) + data[idx1, idy1, idz1] * xd

    c0 = c00 * (1.0 - yd) + c10 * yd
    c1 = c01 * (1.0 - yd) + c11 * yd

    c = c0 * (1.0 - zd) + c1 * zd

    return c
