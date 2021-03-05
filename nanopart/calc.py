import numpy as np
from pathlib import Path

from typing import Tuple

import nanopart
from nanopart.io import read_nanoparticle_file


def calculate_limits(
    responses: np.ndarray, method: str, sigma: float = None, epsilon: float = None
) -> Tuple[str, float, float, float]:

    if responses is None or responses.size == 0:
        return

    mean = np.nanmean(responses)
    gaussian = None
    poisson: Tuple[float, float] = None

    if method == "Automatic":
        method = "Poisson" if mean < 50.0 else "Gaussian"

    if method in ["Highest", "Gaussian"]:
        if sigma is not None:
            gaussian = mean + sigma * np.nanstd(responses)

    if method in ["Highest", "Poisson"]:
        if epsilon is not None:
            yc, yd = nanopart.poisson_limits(mean, epsilon=epsilon)
            poisson = (mean + yc, mean + yd)

    if method == "Highest":
        if gaussian is not None and poisson is not None:
            method = "Gaussian" if gaussian > poisson[1] else "Poisson"

    if method == "Gaussian" and gaussian is not None:
        return (method, mean, gaussian, gaussian)
    elif method == "Poisson" and poisson is not None:
        return (method, mean, poisson[0], poisson[1])
    else:
        return None


def process_file_detections(
    file: Path,
    limit_method: str,
    limit_sigma: float,
    limit_epsilon: float,
    cps_dwelltime: float = None,
) -> dict:
    responses, _ = read_nanoparticle_file(file, delimiter=",")

    # Convert to counts if required
    if cps_dwelltime is not None:
        responses = responses * cps_dwelltime

    size = responses.size

    if responses is None or size == 0:
        raise ValueError(f"Unabled to import file '{file.name}'.")

    limits = calculate_limits(responses, limit_method, limit_sigma, limit_epsilon)

    if limits is None:
        raise ValueError("Limit calculations failed for '{file.name}'.")

    detections, labels = nanopart.accumulate_detections(responses, limits[2], limits[3])
    background = np.nanmean(responses[labels == 0])

    return {
        "file": str(file),
        "events": size,
        "detections": detections,
        "number": detections.size,
        "background": background,
        "limit_method": limits[0],
        "lod": limits[3],
    }


def process_file_mass_response(
    file: Path,
    density: float,
    dwelltime: float,
    molarratio: float,
    massresponse: float,
    limit_method: str = "Automatic",
    limit_sigma: float = 3.0,
    limit_epsilon: float = 0.5,
    response_in_cps: bool = False,
) -> bool:
    result = process_file_detections(
        file,
        limit_method,
        limit_sigma,
        limit_epsilon,
        cps_dwelltime=dwelltime if response_in_cps else None,
    )

    result.update(
        results_from_mass_response(
            result["detections"],
            result["background"],
            result["lod"],
            density=density,
            molarratio=molarratio,
            massresponse=massresponse,
        )
    )

    return result


def process_file_nebulisation_efficiency(
    file: Path,
    density: float,
    dwelltime: float,
    efficiency: float,
    molarratio: float,
    uptake: float,
    response: float,
    time: float,
    limit_method: str = "Automatic",
    limit_sigma: float = 3.0,
    limit_epsilon: float = 0.5,
    response_in_cps: bool = False,
) -> bool:

    result = process_file_detections(
        file,
        limit_method,
        limit_sigma,
        limit_epsilon,
        cps_dwelltime=dwelltime if response_in_cps else None,
    )

    result.update(
        results_from_nebulisation_efficiency(
            result["detections"],
            result["background"],
            result["lod"],
            density=density,
            dwelltime=dwelltime,
            efficiency=efficiency,
            molarratio=molarratio,
            uptake=uptake,
            response=response,
            time=time,
        )
    )

    return result


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
    lod_mass = lod / (massresponse * molarratio)
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
