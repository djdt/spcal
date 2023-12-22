"""Loading data from TOFWERK ICP-ToF."""
from pathlib import Path

import h5py
import numpy as np
import numpy.lib.recfunctions as rfn


def is_tofwerk_file(path: Path) -> bool:
    if not path.exists():
        return False
    if not path.suffix.lower() == ".h5":
        return False
    # Check for TofDAQ Version?
    return True


def calibrate_index_to_mass(
    indices: np.ndarray, mode: int, p: list[float]
) -> np.ndarray:
    """Calibrate sample indicies to mass / charge.

    Args:
        indices: array of sample incidies
        mode: mode from "/FullSpectra/MassCalibMode"
        ps: coefficients from "/FullSpectra/MassCalibration p_"

    Returns:
        calibrated masses
    """

    match mode:
        case 0:  # i = p1 * sqrt(m) + p2
            return np.square((indices - p[1]) / p[0])
        case 1:  # i = p1 / sqrt(m) + p2
            return np.square(p[0] / (indices - p[1]) / p[0])
        case 2:  # i = p1 * m ^ p3 + p2
            return np.power((indices - p[1]) / p[0], 1.0 / p[2])
        case 3:  # i = p1 * sqrt(m) + p2 + p3 * (m - p4) ^ 2
            raise ValueError("perform_mass_calibration: mode 3 not supported.")
        case 4:  # i = p1 * sqrt(m) + p2 + p3 * m ^ 2 + p4 * m + p5
            raise ValueError("perform_mass_calibration: mode 4 not supported.")
        case 5:  # m = p1 * i ^ 2 + p3
            return p[0] * np.square(indices) + p[2]
        case _:
            raise ValueError(f"perform_mass_calibration: unknown mode {mode}.")


def calibrate_mass_to_index(
    masses: np.ndarray, mode: int, p: list[float]
) -> np.ndarray:
    """Calibrate mass / charge to sample indicies.

    Args:
        indices: array of sample incidies
        mode: mode from "/FullSpectra/MassCalibMode"
        ps: coefficients from "/FullSpectra/MassCalibration p_"

    Returns:
        sample indicies
    """

    match mode:
        case 0:  # i = p1 * sqrt(m) + p2
            idx = p[0] * np.sqrt(masses) + p[1]
        case 1:  # i = p1 / sqrt(m) + p2
            idx = p[0] / np.sqrt(masses) + p[1]
        case 2:  # i = p1 * m ^ p3 + p2
            idx = p[0] * np.power(masses, p[2]) + p[1]
        case 3:  # i = p1 * sqrt(m) + p2 + p3 * (m - p4) ^ 2
            raise ValueError("perform_mass_calibration: mode 3 not supported.")
        case 4:  # i = p1 * sqrt(m) + p2 + p3 * m ^ 2 + p4 * m + p5
            raise ValueError("perform_mass_calibration: mode 4 not supported.")
        case 5:  # m = p1 * i ^ 2 + p3
            idx = np.sqrt(masses - p[2] / p[0])
        case _:
            raise ValueError(f"perform_mass_calibration: unknown mode {mode}.")

    return np.around(idx, 0).astype(np.uint32)


def factor_extraction_to_acquisition(h5: h5py._hl.files.File) -> float:
    return float(
        h5.attrs["NbrWaveforms"][0]
        * h5.attrs["NbrBlocks"][0]
        * h5.attrs["NbrMemories"][0]
        * h5.attrs["NbrCubes"][0]
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

    mode = h5["FullSpectra"].attrs["MassCalibMode"][0]
    ps = [
        h5["FullSpectra"].attrs["MassCalibration p1"][0],
        h5["FullSpectra"].attrs["MassCalibration p2"][0],
    ]
    if mode in [2, 5]:
        ps.append(h5["FullSpectra"].attrs["MassCalibration p3"][0])

    lower = calibrate_mass_to_index(
        peak_table["lower integration limit"][idx], mode, ps
    )
    upper = calibrate_mass_to_index(
        peak_table["upper integration limit"][idx], mode, ps
    )
    indicies = np.stack((lower, upper + 1), axis=1)

    peaks = np.empty((*tof_data.shape[:-1], lower.size), dtype=np.float32)
    # This is slow since we need to acces many GB of info.
    for i, sample in enumerate(tof_data):
        peaks[i] = np.add.reduceat(sample, indicies.flat, axis=-1)[..., ::2]

    scale_factor = float(
        (h5["FullSpectra"].attrs["SampleInterval"][0] * 1e9)  # mV * index -> mV * ns
        / h5["FullSpectra"].attrs["Single Ion Signal"][0]  # mV * ns -> ions
        / factor_extraction_to_acquisition(h5)  # ions -> ions/extraction
    )

    return peaks * scale_factor


def read_tofwerk_file(
    path: Path | str, idx: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """Reads a TOFWERK TofDaq .hdf and returns peak data and peak info.

    Args:
        path: path to .hdf archive
        idx: limit extraction to these idx

    Returns:
        structured array of peak data in ions / acquisition
        information from the PeakTable
        dwelltime
    """
    path = Path(path)

    with h5py.File(path, "r") as h5:
        if idx is None:
            idx = np.arange(h5["PeakData"]["PeakTable"].shape[0])
        idx = np.asarray(idx)

        if "PeakData" in h5["PeakData"]:
            data = h5["PeakData"]["PeakData"][..., idx]
        else:
            data = integrate_tof_data(h5, idx=idx)

        data *= factor_extraction_to_acquisition(h5)
        info = h5["PeakData"]["PeakTable"][idx]
        dwell = (
            float(h5["TimingData"].attrs["TofPeriod"][0])
            * 1e-9
            * factor_extraction_to_acquisition(h5)
        )

    names = [x.decode() for x in info["label"]]
    data = rfn.unstructured_to_structured(data.reshape(-1, data.shape[-1]), names=names)

    return data, info, dwell
