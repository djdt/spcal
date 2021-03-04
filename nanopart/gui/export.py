import numpy as np
from pathlib import Path

import nanopart
from nanopart.io import read_nanoparticle_file

from nanopart.calc import (
    calculate_limits,
    results_from_mass_response,
    results_from_nebulisation_efficiency,
)

from typing import Tuple


# def export_results(path: Path,
#             f"Detected particles,{self.sizes.size}\n"
#             f"Number concentration,{self.number.value()},{self.number.unit()}\n"
#             f"Concentration,{self.conc.value()},{self.conc.unit()}\n"
#             f"Ionic background,{self.background.value()},{self.background.unit()}\n"
#             f"Mean NP size,{np.mean(self.sizes) * 1e9},nm\n"
#             f"Median NP size,{np.median(self.sizes) * 1e9},nm\n"
#             f"LOD equivalent size,{self.background_lod_size * 1e9},nm\n"
#         )

#         header = text + "Masses (kg),Sizes (m)"
#         data = np.stack((self.masses, self.sizes), axis=1)

#         np.savetxt(
#             path,
#             data,
#             delimiter=",",
#             header=header,
#         )
# )


def process_file_detections(
    file: Path, limit_method: str, limit_sigma: float, limit_epsilon: float
) -> Tuple[np.ndarray, float, Tuple[str, float, float, float]]:
    responses, _ = read_nanoparticle_file(file, delimiter=",")

    if responses is None or responses.size == 0:
        raise ValueError(f"Unabled to import file '{file.name}'.")

    limits = calculate_limits(responses, limit_method, limit_sigma, limit_epsilon)

    if limits is None:
        raise ValueError("Limit calculations failed for '{file.name}'.")

    detections, labels = nanopart.accumulate_detections(responses, limits[2], limits[3])
    background = np.nanmean(responses[labels == 0])

    return detections, background, limits


def process_file_mass_response(
    file: Path,
    dwelltime: float,
    density: float,
    molarratio: float,
    massresponse: float,
    limit_method: str = "Automatic",
    limit_sigma: float = 3.0,
    limit_epsilon: float = 0.5,
) -> bool:

    responses, _ = read_nanoparticle_file(file, delimiter=",")

    if responses is None or responses.size == 0:
        raise ValueError(f"Unabled to import file '{file.name}'.")

    limits = calculate_limits(responses, limit_method, limit_sigma, limit_epsilon)

    if limits is None:
        raise ValueError("Limit calculations failed for '{file.name}'.")

    detections, labels = nanopart.accumulate_detections(responses, limits[2], limits[3])
    background = np.nanmean(responses[labels == 0])

    return results_from_mass_response(
        detections,
        background,
        limits[3],
        dwelltime=dwelltime,
        density=density,
        molarratio=molarratio,
        massresponse=massresponse,
    )


def process_file_nebulisation_efficiency(
    file: Path,
    dwelltime: float,
    density: float,
    efficiency: float,
    molarratio: float,
    uptake: float,
    response: float,
    time: float,
    limit_method: str = "Automatic",
    limit_sigma: float = 3.0,
    limit_epsilon: float = 0.5,
) -> bool:

    responses, _ = read_nanoparticle_file(file, delimiter=",")

    if responses is None or responses.size == 0:
        raise ValueError(f"Unabled to import file '{file.name}'.")

    limits = calculate_limits(responses, limit_method, limit_sigma, limit_epsilon)

    if limits is None:
        raise ValueError("Limit calculations failed for '{file.name}'.")

    detections, labels = nanopart.accumulate_detections(responses, limits[2], limits[3])
    background = np.nanmean(responses[labels == 0])

    return results_from_nebulisation_efficiency(
        detections,
        background,
        limits[3],
        dwelltime=dwelltime,
        density=density,
        efficiency=efficiency,
        molarratio=molarratio,
        uptake=uptake,
        response=response,
    )
