"""Misc and helper calculation functions."""
from bisect import bisect_left, insort
import numpy as np

try:
    import bottleneck as bn

    bottleneck_found = True
except ImportError:
    bottleneck_found = False


import spcal
from spcal.poisson import formula_c as poisson_limits

from typing import Dict, Tuple


def moving_mean(x: np.ndarray, n: int) -> np.ndarray:
    """Calculates the rolling mean.

    Uses bottleneck.move_mean if available otherwise np.cumsum based algorithm.

    Args:
        x: array
        n: window size
    """
    if bottleneck_found:
        return bn.move_mean(x, n)[n - 1 :]
    r = np.cumsum(x)
    r[n:] = r[n:] - r[:-n]  # type: ignore
    return r[n - 1 :] / n


def moving_median(x: np.ndarray, n: int) -> np.ndarray:
    """Calculates the rolling median.

    Uses bottleneck.move_median if available otherwise sort based algorithm.

    Args:
        x: array
        n: window size
    """
    if bottleneck_found:
        return bn.move_median(x, n)[n - 1 :]

    r = np.empty(x.size - n + 1, x.dtype)
    sort = sorted(x[:n])
    m = n // 2
    m2 = m + n % 2 - 1

    for start in range(x.size - n):
        r[start] = sort[m] + sort[m2]
        end = start + n
        del sort[bisect_left(sort, x[start])]
        insort(sort, x[end])

    r[-1] = sort[m] + sort[m2]
    return r / 2.0


def moving_std(x: np.ndarray, n: int) -> np.ndarray:
    """Calculates the rolling standard deviation.

    Uses bottleneck.move_std if available otherwise np.cumsum based algorithm.

    Args:
        x: array
        n: window size
    """
    if bottleneck_found:
        return bn.move_std(x, n)[n - 1 :]

    sums = np.empty(x.size - n + 1)
    sqrs = np.empty(x.size - n + 1)

    tab = np.cumsum(x) / n
    sums[0] = tab[n - 1]
    sums[1:] = tab[n:] - tab[:-n]

    tab = np.cumsum(x * x) / n
    sqrs[0] = tab[n - 1]
    sqrs[1:] = tab[n:] - tab[:-n]

    return np.sqrt(sqrs - sums * sums)


def calculate_limits(
    responses: np.ndarray,
    method: str,
    sigma: float = 3.0,
    error_rates: Tuple[float, float] = (0.05, 0.05),
    window: int | None = None,
) -> Tuple[str, Dict[str, float], np.ndarray]:
    """Calculates limit(s) of detections for input.

    If `window` is given then rolling filters are used to create limits. The returned values are
    then arrays the same size as `responses`.

    `method` 'Automatic' will return 'Gaussian' if mean(responses) > 50.0, otherwise 'Poisson'.
    `method` 'Highest' will return the maximum of 'Gaussian' and 'Poisson'.
    'Gaussian' is calculated as mean(responses) + `sigma` * std(responses).
    'Poisson' uses `:func:spcal.poisson.formula_c`.

    Args:
        responses: array of signals
        method: method to use {'Automatic', 'Highest', 'Gaussian', 'Gaussian Median', 'Poisson'}
        sigma: threshold term for 'Gaussian'
        error_rates: α and β rates for 'Poisson'
        window: rolling limits

    Returns:
        method, {symbol: value, ...}, array[('mean', 'lc', 'ld')]
    """
    if responses is None or responses.size == 0:
        raise ValueError("Responses invalid.")

    if method not in ["Automatic", "Highest", "Gaussian", "Gaussian Median", "Poisson"]:
        raise ValueError(
            'method must be one of "Automatic", "Highest", "Gaussian", "Gaussian Median", "Poisson"'
        )

    if "Median" in method:
        ub = np.median(responses)
    else:
        ub = np.mean(responses)

    assert isinstance(ub, float)

    limit_params = {}

    if method == "Automatic":
        method = "Poisson" if ub < 50.0 else "Gaussian"
    elif method == "Highest":
        lpoisson = ub + poisson_limits(ub, alpha=error_rates[0], beta=error_rates[1])[1]
        lgaussian = ub + sigma * np.std(responses)
        method = "Gaussian" if lgaussian > lpoisson else "Poisson"

    if window is None or window < 2:
        limits = np.array([(ub, 0.0, 0.0)], dtype=calculate_limits.dtype)
        if "Gaussian" in method:
            std = np.std(responses)
    else:
        pad = np.pad(responses, [window // 2, window // 2], mode="reflect")
        limits = np.empty(responses.size, dtype=calculate_limits.dtype)

        if "Median" in method:
            limits["mean"] = moving_median(pad, window)[: responses.size]
        else:
            limits["mean"] = moving_mean(pad, window)[: responses.size]

        if "Gaussian" in method:
            std = moving_std(pad, window)[: responses.size]

    if "Gaussian" in method:
        limits["ld"] = limits["mean"] + sigma * std
        limits["lc"] = limits["ld"]
        limit_params["σ"] = sigma
    else:
        sc, sd = poisson_limits(
            limits["mean"], alpha=error_rates[0], beta=error_rates[1]
        )
        limits["lc"] = limits["mean"] + sc
        limits["ld"] = limits["mean"] + sd
        limit_params["α"] = error_rates[0]
        limit_params["β"] = error_rates[1]

    return method, limit_params, limits


calculate_limits.dtype = np.dtype(
    {"names": ["mean", "lc", "ld"], "formats": [np.float64, np.float64, np.float64]}
)


def results_from_mass_response(
    detections: np.ndarray,
    background: float,
    lod: np.ndarray,
    density: float,
    mass_fraction: float,
    mass_response: float,
) -> dict:
    """Calculates the masses, sizes and lods from mass response.

    All values are in SI units.
    The `lod` should be calculated by `:func:spcal.calculate_limits`.

    Args:
        detections: array of summed signals
        background: background mean
        lod: limit of detection(s)
        density: of NP (kg/m3)
        massfraction: of NP
        massresponse: of a reference material (kg/count)

    Returns:
        dict of results
    """

    # if isinstance(lod, np.ndarray):
    #     lod = np.array([np.amin(lod), np.amax(lod), np.mean(lod), np.median(lod)])

    masses = detections * (mass_response / mass_fraction)
    sizes = spcal.particle_size(masses, density=density)

    bed = spcal.particle_size(
        background * (mass_response / mass_fraction), density=density
    )
    lod_mass = lod * (mass_response / mass_fraction)
    lod_size = spcal.particle_size(lod_mass, density=density)

    return {  # type: ignore
        "masses": masses,
        "sizes": sizes,
        "background_size": bed,
        "lod": lod,
        "lod_mass": lod_mass,
        "lod_size": lod_size,
    }


def results_from_nebulisation_efficiency(
    detections: np.ndarray,
    background: float,
    lod: np.ndarray,
    density: float,
    dwelltime: float,
    efficiency: float,
    mass_fraction: float,
    uptake: float,
    response: float,
    time: float,
) -> dict:
    """Calculates the masses, sizes, background and lods from transport efficiency.

    All values are in SI units.
    The `lod` should be calculated by `:func:spcal.calculate_limits`.

    Args:
        detections: array of summed signals
        background: background mean
        lod: limit of detection(s)
        density: of NP (kg/m3)
        dwelltime: quadrupole dwell time (s)
        efficiency: transport efficiency
        massfraction: of NP
        uptake: sample flow rate (L/s)
        response: of an ionic standard (count/(kg/L))
        time: total aquisition time (s)

    Returns:
        dict of results
    """

    # if isinstance(lod, np.ndarray):
    #     lod = np.array([np.amin(lod), np.amax(lod), np.mean(lod), np.median(lod)])

    masses = spcal.particle_mass(
        detections,
        dwell=dwelltime,
        efficiency=efficiency,
        flow_rate=uptake,
        response_factor=response,
        mass_fraction=mass_fraction,
    )
    sizes = spcal.particle_size(masses, density=density)

    number_concentration = np.around(
        spcal.particle_number_concentration(
            detections.size,
            efficiency=efficiency,
            flow_rate=uptake,
            time=time,
        )
    )
    concentration = spcal.particle_total_concentration(
        masses,
        efficiency=efficiency,
        flow_rate=uptake,
        time=time,
    )

    ionic = background / response
    bed = spcal.particle_size(
        spcal.particle_mass(
            background,
            dwell=dwelltime,
            efficiency=efficiency,
            flow_rate=uptake,
            response_factor=response,
            mass_fraction=mass_fraction,
        ),
        density=density,
    )
    lod_mass = spcal.particle_mass(
        lod,
        dwell=dwelltime,
        efficiency=efficiency,
        flow_rate=uptake,
        response_factor=response,
        mass_fraction=mass_fraction,
    )
    lod_size = spcal.particle_size(lod_mass, density=density)

    return {
        "masses": masses,
        "sizes": sizes,
        "concentration": concentration,
        "number_concentration": number_concentration,
        "background_concentration": ionic,
        "background_size": bed,
        "lod": lod,
        "lod_mass": lod_mass,
        "lod_size": lod_size,
    }
