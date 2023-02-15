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


def perform_mass_calibration(
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


def read_tofwerk_file(path: Path | str) -> Tuple[np.ndarray, np.ndarray]:
    """Reads a TOFWERK TofDaq .hdf and returns peak data and peak info.

    Args:
        path: path to .hdf archive

    Returns:
        structured array of peak data
        information from the PeakTable
    """
    path = Path(path)

    with h5py.File(path, "r") as fp:
        data = fp["PeakData"]["PeakData"][:]
        info = fp["PeakData"]["PeakTable"][:]
        extraction_time = fp["TimingData"].attrs["TofPeriod"] * 1e-9  # in ns
        # Needed?
        sis = fp["FullSpectra"].attrs["Single Ion Signal"]

    data /= (extraction_time * sis)  # cps -> counts
    names = [x.decode() for x in info["label"]]
    data = rfn.unstructured_to_structured(data.reshape(-1, data.shape[-1]), names=names)

    return data, info
