import numpy as np
from statistics import NormalDist

from typing import Tuple, Union

# https://www.epa.gov/radiation/marlap-manual-and-supporting-documents
# https://academic.oup.com/biomet/article/28/3-4/437/220104

def limits(
    ub: Union[float, np.ndarray],
    alpha: float = 0.05,
    beta: float = 0.05,
    epsilon: float = 0.5,
    force_epsilon: bool = False,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Calulate Yc and Yd for mean `ub`.

    Uses a false positive / negative rate of 'alpha' and 'beta'.
    If `ub` if lower than 5.0, the correction factor `epsilon` is added to `ub`.
    Lc and Ld can be calculated by adding `ub` to `Yc` and `Yd`.

    Args:
        ub: mean of background
        alpha: false positive rate
        beta: false negative rate
        epsilon: low `ub` correct factor
        force_epsilon: always use `epsilon`

    Returns:
        Yc, gross count critical value
        Yd, gross count detection limit

    References:
        Currie, L. A. (1968). Limits for qualitative detection and quantitative
            determination. Application to radiochemistry.
            Analytical Chemistry, 40(3), 586–593.
            doi:10.1021/ac60259a007
        Currie, L.A. On the detection of rare, and moderately rare, nuclear events.
            J Radioanal Nucl Chem 276, 285–297 (2008).
            https://doi.org/10.1007/s10967-008-0501-5
    """
    z_a = NormalDist().inv_cdf((1.0 - alpha))
    z_b = NormalDist().inv_cdf((1.0 - beta))

    # 5 counts limit to maintain alpha / beta (Currie 2008)
    if force_epsilon:
        ub = ub + epsilon
    else:
        ub = np.where(ub < 5.0, ub + epsilon, ub)

    # Yc and Yd for paired distribution (Currie 1969)
    Yc = z_a * np.sqrt(2.0 * ub)
    return Yc, np.square(z_b) + 2.0 * Yc
