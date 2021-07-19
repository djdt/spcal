from bisect import bisect_left, insort
import numpy as np

try:
    import bottleneck as bn

    bottleneck_found = True
except ImportError:
    bottleneck_found = False


import nanopart

from typing import Tuple, Union


def moving_mean(x: np.ndarray, n: int) -> np.ndarray:
    if bottleneck_found:
        return bn.move_mean(x, n)[n - 1 :]
    r = np.cumsum(x)
    r[n:] = r[n:] - r[:-n]
    return r[n - 1 :] / n


def moving_median(x: np.ndarray, n: int) -> np.ndarray:
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
            + nanopart.poisson_limits(ub, epsilon=epsilon, force_epsilon=force_epsilon)[
                1
            ]
        )
        lgaussian = ub + sigma * np.std(responses)
        method = "Gaussian" if lgaussian > lpoisson else "Poisson"

    if window is None or window < 2:
        if "Gaussian" in method:
            std = np.std(responses)
            ld = ub + sigma * std
            return ((method, sigma), ub, ld, ld)
        else:
            yc, yd = nanopart.poisson_limits(
                ub, epsilon=epsilon, force_epsilon=force_epsilon
            )
            return ((method, epsilon), ub, ub + yc, ub + yd)
    else:
        pad = np.pad(responses, [window // 2, window // 2], mode="edge")
        if "Median" in method:
            ub = moving_median(pad, window)[: responses.size]
        else:
            ub = moving_mean(pad, window)[: responses.size]

        if "Gaussian" in method:
            std = moving_std(pad, window)[: responses.size]
            ld = ub + sigma * std
            return ((method, sigma), ub, ld, ld)
        else:
            yc, yd = nanopart.poisson_limits(
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
) -> dict:

    if isinstance(lod, np.ndarray):
        lod = np.array([np.amin(lod), np.amax(lod), np.mean(lod), np.median(lod)])

    masses = detections * (massresponse / massfraction)
    sizes = nanopart.particle_size(masses, density=density)

    bed = nanopart.particle_size(
        background * (massresponse / massfraction), density=density
    )
    lod_mass = lod * (massresponse / massfraction)
    lod_size = nanopart.particle_size(lod_mass, density=density)

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
) -> dict:

    if isinstance(lod, np.ndarray):
        lod = np.array([np.amin(lod), np.amax(lod), np.mean(lod), np.median(lod)])

    masses = nanopart.particle_mass(
        detections,
        dwell=dwelltime,
        efficiency=efficiency,
        flowrate=uptake,
        response_factor=response,
        mass_fraction=massfraction,
    )
    sizes = nanopart.particle_size(masses, density=density)

    number_concentration = np.around(
        nanopart.particle_number_concentration(
            detections.size,
            efficiency=efficiency,
            flowrate=uptake,
            time=time,
        )
    )
    concentration = nanopart.particle_total_concentration(
        masses,
        efficiency=efficiency,
        flowrate=uptake,
        time=time,
    )

    ionic = background / response
    bed = nanopart.particle_size(
        nanopart.particle_mass(
            background,
            dwell=dwelltime,
            efficiency=efficiency,
            flowrate=uptake,
            response_factor=response,
            mass_fraction=massfraction,
        ),
        density=density,
    )
    lod_mass = nanopart.particle_mass(
        lod,
        dwell=dwelltime,
        efficiency=efficiency,
        flowrate=uptake,
        response_factor=response,
        mass_fraction=massfraction,
    )
    lod_size = nanopart.particle_size(lod_mass, density=density)

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
