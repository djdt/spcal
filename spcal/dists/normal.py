import numpy as np


def cdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return 0.5 * (1.0 + erf((x - mu) / (sigma * np.sqrt(2.0))))


def pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return 1.0 / (sigma * np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def erf(x: float | np.ndarray) -> float | np.ndarray:
    """Error function approximation.

    The maximum error is 1.5e-7 [1].

    Args:
        x: value

    Returns:
        approximation of error function

    References:
        .. [1] Abramowitz, Milton, and Irene A. Stegun, eds. Handbook of mathematical
            functions with formulas, graphs, and mathematical tables.
            Vol. 55. US Government printing office, 1970.
    """
    t = 1.0 / (1.0 + 0.3275911 * np.abs(x))
    a = np.array([0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429])
    p = np.array([[1, 2, 3, 4, 5]]).T
    e = 1.0 - np.sum(a * np.power(t, p).T, axis=1) * np.exp(-(x**2))
    return np.clip(e, -1.0, 1.0) * np.sign(x)


def erfinv(x: float | np.ndarray) -> float | np.ndarray:
    """The inverse error function.

    Maximum error is 1.5e-9 / sqrt(2).

    Args:
        x: input (-1 - 1)

    Returns:
        inverse error
    """
    return standard_quantile((x + 1.0) / 2.0) / np.sqrt(2.0)


def standard_quantile(p: float | np.ndarray) -> float | np.ndarray:
    """Approximation of the standard normal quantile.

    The maximum error is 1.5e-9 [2]

    Args:
        p: quantile (0 - 1)

    Returns:
        quantile of the standard normal at p

    References:
        .. [2] Dyer, J.S. and Dyer, S.A., 2008. Approximations to inverse error
            functions. IEEE Instrumentation & Measurement Magazine, 11(5), pp.32-36.
    """
    a = np.array(
        [
            -3.969683028665376e1,
            2.209460984245205e2,
            -2.759285104469687e2,
            1.383577518672690e2,
            -3.066479806614716e1,
            2.506628277459239,
        ]
    )
    b = np.array(
        [
            -5.447609879822406e1,
            1.615858368580409e2,
            -1.556989798598866e2,
            6.680131188771972e1,
            -1.328068155288572e1,
            1.0,
        ]
    )

    c = np.array(
        [
            -7.784894002430293e-3,
            -3.223964580411365e-1,
            -2.400758277161838,
            -2.549732539343734,
            4.374664141464968,
            2.938163982698783,
        ]
    )

    d = np.array(
        [
            7.784695709041462e-3,
            3.224671290700398e-1,
            2.445134137142996,
            3.754408661907416,
            1.0,
        ]
    )

    x = np.asanyarray(p)
    P = np.array([[5, 4, 3, 2, 1, 0]]).T

    def R1(z: np.ndarray) -> np.ndarray:
        return np.sum(c * np.power(z, P).T, axis=1) / np.sum(
            d * np.power(z, P[1:]).T, axis=1
        )

    def R2(z: np.ndarray) -> np.ndarray:
        return np.sum(a * np.power(z, P).T, axis=1) / np.sum(
            b * np.power(z, P).T, axis=1
        )

    idx_tail_low = np.logical_and(x > 0, x < 0.02425)
    idx_tail_high = np.logical_and(x > 0.97575, x < 1)
    idx_mid = np.logical_and(x >= 0.02425, x <= 0.97575)

    y = np.empty_like(x)
    y[x == 0] = -np.inf
    y[idx_tail_low] = R1(np.sqrt(-2.0 * np.log(x[idx_tail_low])))
    y[idx_mid] = (x[idx_mid] - 0.5) * R2((x[idx_mid] - 0.5) ** 2)
    y[idx_tail_high] = -R1(np.sqrt(-2.0 * np.log(1.0 - x[idx_tail_high])))
    y[x == 1] = np.inf

    if isinstance(p, float):
        return float(y)
    return y
