"""Poission limits of detection.

See:
https://www.epa.gov/radiation/marlap-manual-and-supporting-documents
https://academic.oup.com/biomet/article/28/3-4/437/220104
"""
import numpy as np
from statistics import NormalDist

from typing import Tuple


def currie(
    ub: float | np.ndarray,
    alpha: float = 0.05,
    beta: float = 0.05,
    eta: float = 2.0,
    epsilon: float = 0.0,
) -> Tuple[float | np.ndarray, float | np.ndarray]:
    """Calculates Sc and Sd for mean background 'ub'.

    For low backgrounds (ub < 10), 'epsilon' of 0.5 is recommended.

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
        Currie, L.A. On the detection of rare, and moderately rare, nuclear events.
            J Radioanal Nucl Chem 276, 285–297 (2008).
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
) -> Tuple[float | np.ndarray, float | np.ndarray]:
    """Calculates Sc and Sd for net background 'Nb'.

    Uses the equations from the MARLAP manual, 20.48, 20.73.
    Reccomended for mean backgrounds > 100.
    Sc equivilent to 'currie(ub=Nb)' if t_sample = t_blank.

    Args:
        nb: mean of background
        alpha: false positive rate
        beta: false negative rate

    Returns:
        Sc, net critical value
        Sd, minimum detection net value

    References:
        United States Environmental Protection Agency,
            MARLAP Manual Volume III: Chapter 20, Detection and Quantification Capabilities Overview
    """
    z_a = NormalDist().inv_cdf((1.0 - alpha))
    z_b = NormalDist().inv_cdf((1.0 - beta))

    tr = t_sample / t_blank

    Sc = z_a * np.sqrt(Nb * tr * (1.0 + tr))
    Sd = Sc + z_b**2 / 2.0 + z_b * np.sqrt((z_b**2 / 4.0) + Sc + Nb * (1.0 + tr))

    return Sc, Sd


def formula_c(
    Nb: float | np.ndarray,
    alpha: float = 0.05,
    beta: float = 0.05,
    t_sample: float = 1.0,
    t_blank: float = 1.0,
) -> Tuple[float | np.ndarray, float | np.ndarray]:
    """Calculates Sc and Sd for net background 'Nb'.

    Uses the equations from the MARLAP manual, 20.52, 20.73.
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
        United States Environmental Protection Agency,
            MARLAP Manual Volume III: Chapter 20, Detection and Quantification Capabilities Overview
    """
    z_a = NormalDist().inv_cdf((1.0 - alpha))
    z_b = NormalDist().inv_cdf((1.0 - beta))

    tr = t_sample / t_blank

    Sc = z_a**2 / 2.0 * tr + z_a * np.sqrt(z_a**2 / 4.0 * tr + Nb * tr * (1.0 + tr))
    Sd = Sc + z_b**2 / 2.0 + z_b * np.sqrt((z_b**2 / 4.0) + Sc + Nb * (1.0 + tr))

    return Sc, Sd


def stapleton_approximation(
    Nb: float | np.ndarray,
    alpha: float = 0.05,
    beta: float = 0.05,
    t_sample: float = 1.0,
    t_blank: float = 1.0,
) -> Tuple[float | np.ndarray, float | np.ndarray]:
    """Calculates Sc and Sd for net background 'Nb'.

    Uses the equations from the MARLAP manual, 20.54, 20.74.
    Reccomended for low mean backgrounds.

    Args:
        nb: mean of background
        alpha: false positive rate
        beta: false negative rate

    Returns:
        Sc, net critical value
        Sd, minimum detection net value

    References:
        United States Environmental Protection Agency,
            MARLAP Manual Volume III: Chapter 20, Detection and Quantification Capabilities Overview
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
