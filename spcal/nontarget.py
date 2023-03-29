import numpy as np
import numpy.lib.recfunctions as rfn

from spcal.limit import SPCalLimit


def non_target_screen(
    x: np.ndarray,
    minimum_count_ppm: float,
    poisson_alpha: float = 0.001,
    gaussian_alpha: float = 1e-6,
) -> np.ndarray:
    drop_names = []
    for name in x.dtype.names:
        limit = SPCalLimit.fromBest(
            x[name], poisson_alpha=poisson_alpha, gaussian_alpha=gaussian_alpha
        )
        count = np.count_nonzero(x[name] > limit)
        if count * 1e6 / x.size < minimum_count_ppm:
            drop_names.append(name)

    return rfn.drop_fields(x, drop_names, usemask=False)
