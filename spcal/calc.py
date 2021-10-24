from bisect import bisect_left, insort
import numpy as np

try:
    import bottleneck as bn

    bottleneck_found = True
except ImportError:
    bottleneck_found = False


import spcal

from typing import Dict, Tuple, Union


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
    r[n:] = r[n:] - r[:-n]
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
    epsilon: float = 0.5,
    force_epsilon: bool = False,
    window: int = None,
) -> Tuple[
    Tuple[str, float],
    Union[float, np.ndarray],
    Union[float, np.ndarray],
    Union[float, np.ndarray],
]:
    """Calculates limit(s) of detections for input.

    If `window` is given then rolling filters are used to create limits. The returned values are
    then arrays the same size as `responses`.

    `method` 'Automatic' will return 'Gaussian' if mean(responses) > 50.0, otherwise 'Poisson'.
    `method` 'Highest' will return the maximum of 'Gaussian' and 'Poisson'.
    'Gaussian' is calculated as mean(responses) + `sigma` * std(responses).
    'Poisson' uses `:func:spcal.poisson_limits`.

    Args:
        responses: array of signals
        method: method to use {'Automatic', 'Highest', 'Gaussian', 'Gaussian Median', 'Poisson'}
        sigma: threshold term for 'Gaussian'
        epsilon: threshold term for 'Poisson'
        force_epsilon: always use `epsilon`
        window: rolling limits

    Returns:
        (method, threshold), mean signal, limit of criticality, limit of detection
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

    if method == "Automatic":
        method = "Poisson" if ub < 50.0 else "Gaussian"
    elif method == "Highest":
        lpoisson = (
            ub
            + spcal.poisson_limits(ub, epsilon=epsilon, force_epsilon=force_epsilon)[1]
        )
        lgaussian = ub + sigma * np.std(responses)
        method = "Gaussian" if lgaussian > lpoisson else "Poisson"

    if window is None or window < 2:
        if "Gaussian" in method:
            std = np.std(responses)
            ld = ub + sigma * std
            return ((method, sigma), ub, ld, ld)
        else:
            yc, yd = spcal.poisson_limits(
                ub, epsilon=epsilon, force_epsilon=force_epsilon
            )
            return ((method, epsilon), ub, ub + yc, ub + yd)
    else:
        pad = np.pad(responses, [window // 2, window // 2], mode="reflect")
        if "Median" in method:
            ub = moving_median(pad, window)[: responses.size]
        else:
            ub = moving_mean(pad, window)[: responses.size]

        if "Gaussian" in method:
            std = moving_std(pad, window)[: responses.size]
            ld = ub + sigma * std
            return ((method, sigma), ub, ld, ld)
        else:
            yc, yd = spcal.poisson_limits(
                ub, epsilon=epsilon, force_epsilon=force_epsilon
            )
            return ((method, epsilon), ub, ub + yc, ub + yd)


def results_from_mass_response(
    detections: np.ndarray,
    background: float,
    lod: Union[float, np.ndarray],
    density: float,
    massfraction: float,
    massresponse: float,
) -> Dict[str, Union[np.ndarray, float]]:
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

    if isinstance(lod, np.ndarray):
        lod = np.array([np.amin(lod), np.amax(lod), np.mean(lod), np.median(lod)])

    masses = detections * (massresponse / massfraction)
    sizes = spcal.particle_size(masses, density=density)

    bed = spcal.particle_size(
        background * (massresponse / massfraction), density=density
    )
    lod_mass = lod * (massresponse / massfraction)
    lod_size = spcal.particle_size(lod_mass, density=density)

    return {
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
    lod: Union[float, np.ndarray],
    density: float,
    dwelltime: float,
    efficiency: float,
    massfraction: float,
    uptake: float,
    response: float,
    time: float,
) -> Dict[str, Union[np.ndarray, float]]:
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

    if isinstance(lod, np.ndarray):
        lod = np.array([np.amin(lod), np.amax(lod), np.mean(lod), np.median(lod)])

    masses = spcal.particle_mass(
        detections,
        dwell=dwelltime,
        efficiency=efficiency,
        flowrate=uptake,
        response_factor=response,
        mass_fraction=massfraction,
    )
    sizes = spcal.particle_size(masses, density=density)

    number_concentration = np.around(
        spcal.particle_number_concentration(
            detections.size,
            efficiency=efficiency,
            flowrate=uptake,
            time=time,
        )
    )
    concentration = spcal.particle_total_concentration(
        masses,
        efficiency=efficiency,
        flowrate=uptake,
        time=time,
    )

    ionic = background / response
    bed = spcal.particle_size(
        spcal.particle_mass(
            background,
            dwell=dwelltime,
            efficiency=efficiency,
            flowrate=uptake,
            response_factor=response,
            mass_fraction=massfraction,
        ),
        density=density,
    )
    lod_mass = spcal.particle_mass(
        lod,
        dwell=dwelltime,
        efficiency=efficiency,
        flowrate=uptake,
        response_factor=response,
        mass_fraction=massfraction,
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
