from pathlib import Path

import h5py
import numpy as np
import pytest

from spcal.io.tofwerk import (
    calibrate_index_to_mass,
    calibrate_mass_to_index,
    factor_extraction_to_acquisition,
    integrate_tof_data,
    is_tofwerk_file,
    read_tofwerk_file,
)


def test_is_tofwerk_file(test_data_path: Path):
    assert is_tofwerk_file(test_data_path.joinpath("tofwerk/tofwerk_au_50nm.h5"))
    assert not is_tofwerk_file(test_data_path.joinpath("text/text_normal.csv"))
    assert not is_tofwerk_file(test_data_path.joinpath("tofwerk/non_existant.h5"))


def test_calibration_modes():
    x = np.arange(1, 10)
    for mode in [0, 1, 2, 5]:
        masses = calibrate_index_to_mass(x, mode, [1.0, -2.0, 3.0])
        idx = calibrate_mass_to_index(masses, mode, [1.0, -2.0, 3.0])
        assert np.all(x == idx)

    for mode in [3, 4, -1]:
        with pytest.raises(ValueError):
            calibrate_index_to_mass(x, mode, [1.0, -2.0, 3.0])
        with pytest.raises(ValueError):
            calibrate_mass_to_index(x, mode, [1.0, -2.0, 3.0])


def test_calibration(test_data_path: Path):
    path = test_data_path.joinpath("tofwerk/tofwerk_au_50nm.h5")
    with h5py.File(path, "r") as h5:
        mass: np.ndarray = h5["FullSpectra"]["MassAxis"][:]  # type: ignore , h5
    idx = np.arange(mass.size)

    # From file, unable to test other modes
    mode = 2
    ps = [2827.69757786, -5221.04079879, 0.49997113]

    mass_to_idx = np.around(calibrate_mass_to_index(mass, mode, ps), 0).astype(int)
    assert np.all(mass_to_idx == idx)

    idx_to_mass = calibrate_index_to_mass(idx, mode, ps)
    assert np.allclose(idx_to_mass, mass)


def test_integrate(test_data_path: Path):
    path = test_data_path.joinpath("tofwerk/tofwerk_au_50nm.h5")
    with h5py.File(path, "r") as h5:
        data = integrate_tof_data(h5)
        data_ar = integrate_tof_data(h5, idx=[45])  # type: ignore
        peak_data: np.ndarray = h5["PeakData"]["PeakData"][:]  # type: ignore , h5

    assert data.shape[-1] == 315
    assert data_ar.shape[-1] == 1

    assert np.allclose(data, peak_data)
    assert np.allclose(data_ar[..., 0], peak_data[..., 45])


def test_factor_extraction_to_acquisition(test_data_path: Path):
    path = test_data_path.joinpath("tofwerk/tofwerk_au_50nm.h5")
    with h5py.File(path, "r") as h5:
        factor = factor_extraction_to_acquisition(h5)
    assert factor == 33


def test_read_tofwerk_file(test_data_path: Path):
    path = test_data_path.joinpath("tofwerk/tofwerk_au_50nm.h5")
    data, info, dwell = read_tofwerk_file(path)

    assert data.shape == (200,)
    assert data.dtype.names is not None
    assert len(data.dtype.names) == 315
    assert np.all(data["[6Li]+"] == 0.0)
    assert np.isclose(data["[40Ar]+"].mean(), 565.63306)
    assert np.isclose(data["[197Au]+"].mean(), 1.5413016)
    assert np.isclose(data["[197Au]+"].max(), 64.280266)

    for label, mass in [
        (b"[6Li]+", 6.01457),  # first
        (b"[40Ar]+", 39.96183),  # ar
        (b"[138Ba]++", 68.95207),  # double charge
        (b"[197Au]+", 196.96602),  # analyte
        (b"UO+", 254.04515),  # last, oxide
    ]:
        assert label in info["label"]
        assert np.isclose(mass, info["mass"][info["label"] == label])

    assert np.isclose(dwell, 0.0009999)
