import numpy as np
from pathlib import Path

import nanopart
from nanopart.io import read_nanoparticle_file

from nanopart.calc import (
    calculate_limits,
    results_from_mass_response,
    results_from_nebulisation_efficiency,
)


def export_results(path: Path, result: dict) -> None:
    with path.open("w") as fp:
        fp.write("# NanoPart Export v0.1\n")
        fp.write(f"# File,{result['file']}\n")
        fp.write(f"# Acquisition events,{result['size']}\n")
        fp.write(f"# Detected particles,{result['number']}\n")
        fp.write(f"# Limit method,{result['limit_method']}\n")
        fp.write(f"# Background,{result['background']},counts\n")
        if "background_size" in result:
            fp.write(f"# Background,{result['background_size']},m\n")
        fp.write(f"# Limit of detection,{result['lod']},counts\n")
        if "lod_mass" in result:
            fp.write(f"Limit of detection,{result['lod_mass']},kg\n")
        if "lod_size" in result:
            fp.write(f"Limit of detection,{result['lod_size']},m\n")
        if "number_concentration" in result:
            fp.write(f"# Number concentration,{result['number_concentration']},#/L\n")
        if "concentration" in result:
            fp.write(f"# Concentration,{result['concentration']},kg/L\n")
        if "background_concentration" in result:
            fp.write(f"# Ionic background,{result['background_concentration']},kg/L\n")
        fp.write(f"# Mean size,{np.mean(result['sizes'])},m\n")
        fp.write(f"# Median size,{np.median(result['sizes'])},m\n")

        fp.write("Signal,Mass (kg),Size (m)\n")

        np.savetxt(
            fp,
            np.stack((result["detections"], result["masses"], result["sizes"]), axis=1),
            delimiter=",",
        )


def _process_file_detections(
    file: Path, limit_method: str, limit_sigma: float, limit_epsilon: float
) -> dict:
    responses, _ = read_nanoparticle_file(file, delimiter=",")

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
) -> bool:
    result = _process_file_detections(file, limit_method, limit_sigma, limit_epsilon)

    result.update(
        results_from_mass_response(
            result["detections"],
            result["background"],
            result["lod"],
            density=density,
            dwelltime=dwelltime,
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
) -> bool:

    result = _process_file_detections(file, limit_method, limit_sigma, limit_epsilon)

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
        )
    )

    return result
