from statistics import NormalDist

import numpy as np

from spcal.poisson import formula_c


def non_target_screen(
    x: np.ndarray,
    minimum_count_ppm: float,
    poisson_alpha: float = 0.001,
    gaussian_alpha: float = 1e-6,
) -> np.ndarray:
    """Screen data for potential NP signals.
    Finds signals with `minimum_count_ppm` ppm points greater than LOD.
    The LOD is calcualted as per `SPCalLimit.fromBest

    Args:
        x: input data shape (events, elements)
        minimum_count_ppm: minimum number of points above limit
        poisson_alpha: alpha for poisson limit
        gaussian_alpha: alpha for gaussian limit

    Returns:
        indices of elements with potential signals
    """
    z = NormalDist().inv_cdf(1.0 - gaussian_alpha)

    means = np.mean(x, axis=0)
    poisson_limits = means + formula_c(means, alpha=poisson_alpha)[0]
    gaussian_limits = means + np.std(x, axis=0) * z
    limits = np.where(means > 10.0, gaussian_limits, poisson_limits)

    counts = np.count_nonzero(x > limits, axis=0)
    idx = counts * 1e5 / x.shape[1] > minimum_count_ppm
    return np.flatnonzero(idx)
