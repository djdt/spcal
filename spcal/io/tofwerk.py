from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import numpy.lib.recfunctions as rfn


def is_tofwerk_file(path: Path) -> bool:
    if not path.suffix.lower() == ".h5":
        return False
    # Check for TofDAQ Version?
    return True


def calibrate_index_to_mass(
    indices: np.ndarray, full_spectra: h5py._hl.group.Group
) -> np.ndarray:
    """Calibrate sample indicies to mass / charge.

    Args:
        indices: array of sample incidies
        full_spectra: '/FullSpectra' group from HDF5

    Returns:
        calibrated masses
    """

    p1 = full_spectra.attrs["MassCalibration p1"]
    p2 = full_spectra.attrs["MassCalibration p2"]
    mode = full_spectra.attrs["MassCalibMode"]
    match mode:
        case 0:  # i = p1 * sqrt(m) + p2
            return np.square((indices - p2) / p1)
        case 1:  # i = p1 / sqrt(m) + p2
            return np.square(p1 / (indices - p2) / p1)
        case 2:  # i = p1 * m ^ p3 + p2
            p3 = full_spectra.attrs["MassCalibration p3"]
            return np.power((indices - p2) / p1, 1.0 / p3)
        case 3:  # i = p1 * sqrt(m) + p2 + p3 * (m - p4) ^ 2
            raise ValueError("perform_mass_calibration: mode 3 not supported.")
        case 4:  # i = p1 * sqrt(m) + p2 + p3 * m ^ 2 + p4 * m + p5
            raise ValueError("perform_mass_calibration: mode 4 not supported.")
        case 5:  # m = p1 * i ^ 2 + p3
            p3 = full_spectra.attrs["MassCalibration p3"]
            return p1 * np.square(indices) + p3
        case _:
            raise ValueError(f"perform_mass_calibration: unknown mode {mode}.")


def calibrate_mass_to_index(
    masses: np.ndarray, full_spectra: h5py._hl.group.Group
) -> np.ndarray:
    """Calibrate sample indicies to mass / charge.

    Args:
        indices: array of sample incidies
        full_spectra: '/FullSpectra' group from HDF5

    Returns:
        calibrated masses
    """

    p1 = full_spectra.attrs["MassCalibration p1"]
    p2 = full_spectra.attrs["MassCalibration p2"]
    mode = full_spectra.attrs["MassCalibMode"]
    match mode:
        case 0:  # i = p1 * sqrt(m) + p2
            return p1 * np.sqrt(masses) + p2
        case 1:  # i = p1 / sqrt(m) + p2
            return p1 / np.sqrt(masses) + p2
        case 2:  # i = p1 * m ^ p3 + p2
            p3 = full_spectra.attrs["MassCalibration p3"]
            return p1 * np.power(masses, p3) + p2
        case 3:  # i = p1 * sqrt(m) + p2 + p3 * (m - p4) ^ 2
            raise ValueError("perform_mass_calibration: mode 3 not supported.")
        case 4:  # i = p1 * sqrt(m) + p2 + p3 * m ^ 2 + p4 * m + p5
            raise ValueError("perform_mass_calibration: mode 4 not supported.")
        case 5:  # m = p1 * i ^ 2 + p3
            p3 = full_spectra.attrs["MassCalibration p3"]
            return np.sqrt(masses - p3 / p1)
        case _:
            raise ValueError(f"perform_mass_calibration: unknown mode {mode}.")


def factor_extraction_to_acquisition(h5: h5py._hl.files.File) -> float:
    return float(
        h5.attrs["NbrWaveforms"]
        * h5.attrs["NbrBlocks"]
        * h5.attrs["NbrMemories"]
        * h5.attrs["NbrCubes"]
    )


def integrate_tof_data(
    h5: h5py._hl.files.File, idx: np.ndarray | None = None
) -> np.ndarray:
    """Integrates TofData to recreate PeakData.
    Returned data is in ions/extraction for compatibility with PeakData, it can be
    converted to ions/acquisition by via * `factor_extraction_to_acquisition`.
    Integration is summing from int(lower index limit) + 1 to int(upper index limit).

    Args:
        h5: opened h5 file
        idx: only integrate these peak idx

    Returns:
        data equivilent to PeakData
    """
    tof_data = h5["FullSpectra"]["TofData"]
    peak_table = h5["PeakData"]["PeakTable"]

    if idx is None:
        idx = np.arange(peak_table.shape[0])
    idx = np.asarray(idx)

    scale_factor = float(
        (h5["FullSpectra"].attrs["SampleInterval"] * 1e9)  # mV * index -> mV * ns
        / h5["FullSpectra"].attrs["Single Ion Signal"]  # mV * ns -> ions
        / factor_extraction_to_acquisition(h5)  # ions -> ions/extraction
    )

    lower = calibrate_mass_to_index(
        peak_table["lower integration limit"][idx], h5["FullSpectra"]
    )
    upper = calibrate_mass_to_index(
        peak_table["upper integration limit"][idx], h5["FullSpectra"]
    )
    indicies = np.stack((lower + 1, upper), axis=1).astype(np.uint32)

    peaks = np.empty((*tof_data.shape[:-1], lower.size), dtype=np.float32)
    # This is slow since we need to acces many GB of info.
    for i, sample in enumerate(tof_data):
        peaks[i] = np.add.reduceat(sample, indicies.flat, axis=-1)[..., ::2]

    return peaks * scale_factor


def read_tofwerk_file(path: Path | str) -> Tuple[np.ndarray, np.ndarray]:
    """Reads a TOFWERK TofDaq .hdf and returns peak data and peak info.

    Args:
        path: path to .hdf archive

    Returns:
        structured array of peak data in ions / acquisition
        information from the PeakTable
    """
    path = Path(path)

    with h5py.File(path, "r") as h5:
        data = h5["PeakData"]["PeakData"][()]
        data *= factor_extraction_to_acquisition(h5)
        info = h5["PeakData"]["PeakTable"][()]

    names = [x.decode() for x in info["label"]]
    data = rfn.unstructured_to_structured(data.reshape(-1, data.shape[-1]), names=names)

    return data, info