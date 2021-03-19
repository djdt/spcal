import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from typing import Tuple

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

import nanopart

from typing import Callable


def centered_sliding_view(x: np.ndarray, window: int) -> np.ndarray:
    x_pad = np.pad(x, [window // 2, window // 2 - 1], mode="edge")
    view = sliding_window_view(x_pad, window)
    return view


def threaded_over_axis(x: np.ndarray, func: Callable, axis: int = 1) -> np.ndarray:
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(np.mean, y, axis=axis)
            for y in np.array_split(x, cpu_count())
        ]
    return np.concatenate([f.result() for f in futures])


def calculate_limits(
    responses: np.ndarray,
    method: str,
    sigma: float = None,
    epsilon: float = None,
    window: int = None,
) -> Tuple[str, float, float, float]:

    if responses is None or responses.size == 0:
        return

    ub = np.mean(responses)
    if window is None or window < 2:
        mean = ub
    else:
        view = centered_sliding_view(responses, window)
        if "Median" in method:
            mean = threaded_over_axis(view, np.median, axis=1)
        else:
            mean = threaded_over_axis(view, np.mean, axis=1)

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
            std = threaded_over_axis(view, np.std, axis=1)
        gaussian = mean + sigma * std
        return (method, mean, gaussian, gaussian)
    elif "Poisson" in method and epsilon is not None:
        yc, yd = nanopart.poisson_limits(mean, epsilon=epsilon)
        return (method, mean, mean + yc, mean + yd)

    return None


def results_from_mass_response(
    detections: np.ndarray,
    background: float,
    lod: float,
    density: float,
    molarratio: float,
    massresponse: float,
) -> dict:

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
        "lod_mass": lod_mass,
        "lod_size": lod_size,
    }


def results_from_nebulisation_efficiency(
    detections: np.ndarray,
    background: float,
    lod: float,
    density: float,
    dwelltime: float,
    efficiency: float,
    molarratio: float,
    uptake: float,
    response: float,
    time: float,
) -> dict:

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
        "lod_mass": lod_mass,
        "lod_size": lod_size,
    }
