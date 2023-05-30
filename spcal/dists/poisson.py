import numpy as np


def pdf(k: np.ndarray, lam: float) -> np.ndarray:
    assert np.issubdtype(k.dtype, np.integer)
    assert np.all(k >= 0)

    # There will be overflows for high lam during factorial and power
    with np.errstate(over="ignore"):
        fk = np.cumprod(np.where(k == 0, 1, k), dtype=np.float64)
        pdf = np.zeros(k.size, dtype=np.float64)
        np.divide(lam**k * np.exp(-lam), fk, where=np.isfinite(fk), out=pdf)
    return pdf


def cdf(k: np.ndarray, lam: float) -> np.ndarray:
    assert np.issubdtype(k.dtype, np.integer)
    assert np.all(k >= 0)

    j = np.arange(0, np.amax(k) + 1)
    return np.cumsum(pdf(j, lam))[k]
