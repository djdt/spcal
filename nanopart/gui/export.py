import numpy as np
from pathlib import Path

import nanopart

from nanopart.calc import (
    calculate_limits,
    results_from_mass_response,
    results_from_nebulisation_efficiency,
)
from nanopart.io import read_nanoparticle_file, export_nanoparticle_results


def _process_file_detections(
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
    result = _process_file_detections(
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

    result = _process_file_detections(
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
