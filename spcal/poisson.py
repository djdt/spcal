"""Poission limits of criticality and detection.

Formulas used here are taken from the `MARLAP manual
<https://www.epa.gov/radiation/marlap-manual-and-supporting-documents>`_.

The limit of crticality should be used for all particle detection descisions,
not the limit of detection. For an explaination of why, see the manual linked above.

"""

from statistics import NormalDist

import numpy as np


def currie(
    ub: float | np.ndarray,
    alpha: float = 0.05,
    beta: float = 0.05,
    eta: float = 2.0,
    epsilon: float = 0.0,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Calculates Sc and Sd for mean background 'ub'.

    For low backgrounds (ub < 10), 'epsilon' of 0.5 is recommended [1]_.

    Args:
        nb: mean of background
        alpha: false positive rate
        beta: false negative rate
        eta: (r+1)/r, where r is number of background replicates
        epsilon: correction term for low backgrounds

    Returns:
        Sc, net critical value
        Sd, minimum detection net value

    References:
        .. [1] Currie, L.A. On the detection of rare, and moderately rare,
            nuclear events. J Radioanal Nucl Chem 276, 285–297 (2008).
            https://doi.org/10.1007/s10967-008-0501-5
    """

    z_a = NormalDist().inv_cdf((1.0 - alpha))
    z_b = NormalDist().inv_cdf((1.0 - beta))

    Sc = z_a * np.sqrt((ub + epsilon) * eta)
    Sd = z_b**2 + 2.0 * Sc

    return Sc, Sd


def formula_a(
    Nb: float | np.ndarray,
    alpha: float = 0.05,
    beta: float = 0.05,
    t_sample: float = 1.0,
    t_blank: float = 1.0,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Calculates Sc and Sd for net background 'Nb'.

    Uses the equations from the MARLAP manual, 20.48, 20.73 [2]_.
    Reccomended for mean backgrounds > 100.
    Sc equivilent to ``currie(ub=Nb)`` if t_sample = t_blank.

    Args:
        nb: mean of background
        alpha: false positive rate
        beta: false negative rate

    Returns:
        Sc, net critical value
        Sd, minimum detection net value

    References:
        .. [2] United States Environmental Protection Agency,
            MARLAP Manual Volume III: Chapter 20,
            Detection and Quantification Capabilities Overview
    """
    z_a = NormalDist().inv_cdf((1.0 - alpha))
    z_b = NormalDist().inv_cdf((1.0 - beta))

    tr = t_sample / t_blank

    Sc = z_a * np.sqrt(Nb * tr * (1.0 + tr))
    Sd = Sc + z_b**2 / 2.0 + z_b * np.sqrt((z_b**2 / 4.0) + Sc + Nb * (1.0 + tr))

    return Sc, Sd


def formula_b(
    Nb: float | np.ndarray,
    alpha: float = 0.05,
    beta: float = 0.05,
    t_sample: float = 1.0,
    t_blank: float = 1.0,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    raise NotImplementedError(  # pragma: no cover
        "formula_b is not reccomended and is therefore not implemented"
    )


def formula_c(
    Nb: float | np.ndarray,
    alpha: float = 0.05,
    beta: float = 0.05,
    t_sample: float = 1.0,
    t_blank: float = 1.0,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Calculates Sc and Sd for net background 'Nb'.

    Uses the equations from the MARLAP manual, 20.52, 20.73 [3]_.
    Reccomended for low mean backgrounds.
    Sc equivilent to 'decision threshold' of ISO 11929-1.

    Args:
        nb: mean of background
        alpha: false positive rate
        beta: false negative rate

    Returns:
        Sc, net critical value
        Sd, minimum detection net value

    References:
        .. [3] United States Environmental Protection Agency,
            MARLAP Manual Volume III: Chapter 20,
            Detection and Quantification Capabilities Overview
    """
    z_a = NormalDist().inv_cdf((1.0 - alpha))
    z_b = NormalDist().inv_cdf((1.0 - beta))

    tr = t_sample / t_blank

    Sc = z_a**2 / 2.0 * tr + z_a * np.sqrt(
        z_a**2 / 4.0 * tr * tr + Nb * tr * (1.0 + tr)
    )
    Sd = Sc + z_b**2 / 2.0 + z_b * np.sqrt((z_b**2 / 4.0) + Sc + Nb * (1.0 + tr))

    return Sc, Sd


def stapleton_approximation(
    Nb: float | np.ndarray,
    alpha: float = 0.05,
    beta: float = 0.05,
    t_sample: float = 1.0,
    t_blank: float = 1.0,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Calculates Sc and Sd for net background 'Nb'.

    Uses the equations from the MARLAP manual, 20.54, 20.74 [4]_.
    Reccomended for low mean backgrounds.

    Args:
        nb: mean of background
        alpha: false positive rate
        beta: false negative rate

    Returns:
        Sc, net critical value
        Sd, minimum detection net value

    References:
        .. [4] United States Environmental Protection Agency,
            MARLAP Manual Volume III: Chapter 20,
            Detection and Quantification Capabilities Overview
    """
    z_a = NormalDist().inv_cdf((1.0 - alpha))
    z_b = NormalDist().inv_cdf((1.0 - beta))

    d = z_a / 4.112
    tr = t_sample / t_blank
    rb = Nb / t_blank

    Sc = (
        d * (tr - 1.0)
        + z_a**2 / 4.0 * (1.0 + tr)
        + z_a * np.sqrt((Nb + d) * tr * (1.0 + tr))
    )
    Sd = (z_a + z_b) ** 2 / 4.0 * (1.0 + tr) + (z_a + z_b) * np.sqrt(
        rb * t_sample * (1.0 + tr)
    )

    return Sc, Sd
