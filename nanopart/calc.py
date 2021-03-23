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
        return bn.move_mean(x, n)[n - 1:]
    r = np.cumsum(x)
    r[n:] = r[n:] - r[:-n]
    return r[n - 1 :] / n


def moving_median(x: np.ndarray, n: int) -> np.ndarray:
    if bottleneck_found:
        return bn.move_median(x, n)[n - 1:]
    view = np.lib.stride_tricks.sliding_window_view(x, n)
    return np.median(view, axis=1)


def moving_std(x: np.ndarray, n: int) -> np.ndarray:
    if bottleneck_found:
        return bn.move_std(x, n)[n - 1:]
    view = np.lib.stride_tricks.sliding_window_view(x, n)
    return np.std(view, axis=1)


def calculate_limits(
    responses: np.ndarray,
    method: str,
    sigma: float = None,
    epsilon: float = None,
    window: int = None,
) -> Tuple[
    str, Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]
]:

    if responses is None or responses.size == 0:
        return

    ub = np.mean(responses)
    if window is None or window < 2:
        mean = ub
    else:
        pad = np.pad(responses, [window // 2, window // 2], mode="edge")
        if "Median" in method:
            mean = moving_median(pad, window)[:responses.size]
        else:
            mean = moving_mean(pad, window)[:responses.size]

    if method == "Automatic":
        method = "Poisson" if ub < 50.0 else "Gaussian"
    elif method == "Highest" and sigma is not None and epsilon is not None:
        lpoisson = ub + nanopart.poisson_limits(ub, epsilon=epsilon)[1]
        lgaussian = ub + sigma * np.nanstd(responses)
        method = "Gaussian" if lgaussian > lpoisson else "Poisson"

    if "Gaussian" in method and sigma is not None:
        if window is None or window < 2:
            std = np.std(responses)
        else:
            std = moving_std(pad, window)[:responses.size]
        gaussian = mean + sigma * std
        return (method, mean, gaussian, gaussian)
    elif "Poisson" in method and epsilon is not None:
        yc, yd = nanopart.poisson_limits(mean, epsilon=epsilon)
        return (method, mean, mean + yc, mean + yd)

    return None


def results_from_mass_response(
    detections: np.ndarray,
    background: float,
    lod: Union[float, np.ndarray],
    density: float,
    molarratio: float,
    massresponse: float,
) -> dict:

    if isinstance(lod, np.ndarray):
        lod = np.array([np.amin(lod), np.amax(lod), np.mean(lod), np.median(lod)])

    masses = detections * (massresponse / molarratio)
    sizes = nanopart.particle_size(masses, density=density)

    bed = nanopart.particle_size(
        background * (massresponse / molarratio), density=density
    )
    lod_mass = lod * (massresponse / molarratio)
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
    molarratio: float,
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
        molar_ratio=molarratio,
    )
    sizes = nanopart.particle_size(masses, density=density)

    number_concentration = nanopart.particle_number_concentration(
        detections.size,
        efficiency=efficiency,
        flowrate=uptake,
        time=time,
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
            molar_ratio=molarratio,
        ),
        density=density,
    )
    lod_mass = nanopart.particle_mass(
        lod,
        dwell=dwelltime,
        efficiency=efficiency,
        flowrate=uptake,
        response_factor=response,
        molar_ratio=molarratio,
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
