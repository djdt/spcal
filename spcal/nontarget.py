"""Functions for screening data for interesting signals."""
from statistics import NormalDist

import numpy as np

from spcal.poisson import formula_c


def non_target_screen(
    x: np.ndarray,
    minimum_count_ppm: float,
    thresholds: np.ndarray | None = None,
    poisson_alpha: float = 0.001,
    gaussian_alpha: float = 1e-6,
    gaussian_mean: float = 10.0,
) -> np.ndarray:
    """Screen data for potential NP signals.

    Finds signals with ``minimum_count_ppm`` ppm points greater than threshold.
    If detection thresholds are non passed then thresholds are calculated using
    Formula C from the MARLAP manual or Gaussian statistics if the signal mean is
    above ``gaussian_mean``.

    Args:
        x: data of shape (events, elements)
        minimum_count_ppm: minimum number of points above limit
        thresholds: pre-calculated threshold, shape (elements,)
        poisson_alpha: alpha for Poisson limit
        gaussian_alpha: alpha for Gaussian limit
        gaussian_mean: mean signal above which to switch to Gaussian limit

    Returns:
        indices of elements with potential signals
    """
    z = NormalDist().inv_cdf(1.0 - gaussian_alpha)

    if thresholds is None:
        means = np.mean(x, axis=0)
        sc, _ = formula_c(means, alpha=poisson_alpha)
        poisson_thresholds = (means + sc).astype(int) + 1.0
        gaussian_thresholds = means + np.std(x, axis=0) * z
        thresholds = np.where(
            means > gaussian_mean, gaussian_thresholds, poisson_thresholds
        )

    counts = np.count_nonzero(x > thresholds, axis=0)
    idx = counts * 1e6 / x.shape[0] > minimum_count_ppm
    return np.flatnonzero(idx)
