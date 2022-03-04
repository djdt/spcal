import numpy as np
from statistics import NormalDist

from typing import Tuple, Union

# https://www.epa.gov/radiation/marlap-manual-and-supporting-documents
# https://academic.oup.com/biomet/article/28/3-4/437/220104


def formula_a(
    nb: Union[float, np.ndarray], alpha: float = 0.05, beta: float = 0.05
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Calculates Sc and Sd for mean background 'nb'.

    Uses the equations from the MARLAP manual, 20.54, 20.74, tb=ts is assumed.
    Reccomended for mean backgrounds > 100.

    Args:
        nb: mean of background
        alpha: false positive rate
        beta: false negative rate

    Returns:
        Sc, net critical value
        Sd, minimum detection net value

    References:
        Currie, L.A. On the detection of rare, and moderately rare, nuclear events.
            J Radioanal Nucl Chem 276, 285–297 (2008).
            https://doi.org/10.1007/s10967-008-0501-5

        United States Environmental Protection Agency,
            MARLAP Manual Volume III: Chapter 20, Detection and Quantification Capabilities Overview
    """
    z_a = NormalDist().inv_cdf((1.0 - alpha))
    z_b = NormalDist().inv_cdf((1.0 - beta))

    Sc = z_a * np.sqrt(nb * 2.0)
    Sd = Sc + z_b**2 / 2.0 + z_b * np.sqrt((z_b**2 / 4.0) + Sc + nb * 2.0)

    return Sc, Sd


def stapleton_approximation(
    nb: Union[float, np.ndarray], alpha: float = 0.05, beta: float = 0.05
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Calculates Sc and Sd for mean background 'nb'.

    Uses the equations from the MARLAP manual, 20.54, 20.74, tb=ts is assumed.
    Reccomended for low mean backgrounds.

    Args:
        nb: mean of background
        alpha: false positive rate
        beta: false negative rate

    Returns:
        Sc, net critical value
        Sd, minimum detection net value

    References:
        Currie, L.A. On the detection of rare, and moderately rare, nuclear events.
            J Radioanal Nucl Chem 276, 285–297 (2008).
            https://doi.org/10.1007/s10967-008-0501-5

        United States Environmental Protection Agency,
            MARLAP Manual Volume III: Chapter 20, Detection and Quantification Capabilities Overview
    """
    z_a = NormalDist().inv_cdf((1.0 - alpha))
    z_b = NormalDist().inv_cdf((1.0 - beta))

    d = z_a / 4.112

    Sc = z_a**2 / 2.0 + z_a * np.sqrt((nb + d) * 2.0)

    Sd = (z_a + z_b) ** 2 / 2.0 + (z_a + z_b) * np.sqrt(nb * 2.0)
    return Sc, Sd
