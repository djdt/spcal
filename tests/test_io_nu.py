from pathlib import Path

import numpy as np
import pytest

from spcal.io.nu import (
    acquisition_indicies,
    acquisition_times,
    eventtime_from_info,
    is_nu_directory,
    is_nu_run_info_file,
    read_directory,
    read_integ_binary,
    select_nu_signals,
)


def test_is_nu_run_info_file(test_data_path: Path):
    path = test_data_path.joinpath("nu/normal")
    assert not is_nu_run_info_file(path)
    assert is_nu_run_info_file(path.joinpath("run.info"))


def test_is_nu_dir(test_data_path: Path):
    path = test_data_path.joinpath("nu/normal")
    assert is_nu_directory(path)
    assert not is_nu_directory(path.parent)
    assert not is_nu_directory(path.joinpath("fake"))


def test_acquisition_indicies():
    info = {
        "NumAccumulations1": 2,
        "NumAccumulations2": 4,
        "SegmentInfo": [{"AcquisitionCount": 5049672}, {"AcquisitionCount": 5049672}],
    }
    idx = acquisition_indicies(
        np.array([1, 1, 1, 2]), np.array([1, 1, 2, 1]), np.array([8, 16, 8, 24]), info
    )
    assert idx[0] == 1
    assert idx[1] == 2
    assert idx[2] == (5049672 + 8) // 8
    assert idx[3] == (5049672 * 2 + 24) // 8


def test_acquisition_times():
    info = {
        "NumAccumulations1": 2,
        "NumAccumulations2": 4,
        "SegmentInfo": [
            {"AcquisitionCount": 5049672, "AcquisitionPeriodNs": 0.1},
            {"AcquisitionCount": 5049672, "AcquisitionPeriodNs": 0.2},
        ],
    }
    times = acquisition_times(
        np.array([1, 1, 1, 2]), np.array([1, 1, 2, 1]), np.array([8, 16, 8, 24]), info
    )
    assert times[0] == 0.8
    assert times[1] == 1.6
    assert times[2] == 5049672 * 0.1 + 8 * 0.2
    assert times[3] == 5049672 * 0.1 + 5049672 * 0.2 + 24 * 0.1


def test_io_nu_import(test_data_path: Path):
    path = test_data_path.joinpath("nu/normal")
    masses, signals, _, info, _ = read_directory(path, cycle=1, segment=1)
    assert masses.size == 127
    assert signals.shape == (40, 127)
    assert np.isclose(masses[0], 80.90475)
    assert np.isclose(masses[-1], 208.9560)

    assert np.all(np.isclose(info["MassCalCoefficients"], [-0.221135, 0.000613]))
    assert len(info["SegmentInfo"]) == 1
    assert info["SegmentInfo"][0]["AcquisitionTriggerDelayNs"] == 14950.0

    assert np.isclose(eventtime_from_info(info), 0.09824e-3)


def test_io_nu_import_compressed(test_data_path: Path):
    path = test_data_path.joinpath("nu/compressed")
    masses, signals, _times, info, _ = read_directory(path)
    assert masses.size == 29
    assert signals.shape == (51725, 29)

    assert np.isclose(np.mean(signals[:, 0]), 0.03979577)

    assert np.isclose(eventtime_from_info(info), 88.08e-6)


def test_io_nu_import_integ_limits(test_data_path: Path):
    path = test_data_path.joinpath("nu/normal")
    masses, signals, times, _, _ = read_directory(
        path, last_integ_file=1, cycle=1, segment=1
    )
    assert masses.size == 127
    assert times.shape == (10,)
    assert signals.shape == (10, 127)

    masses, signals, times, _, _ = read_directory(
        path, first_integ_file=1, last_integ_file=3, cycle=1, segment=1
    )
    assert masses.size == 127
    assert times.shape == (20,)
    assert signals.shape == (20, 127)

    with pytest.raises(ValueError):
        read_directory(path, first_integ_file=1, last_integ_file=1, cycle=1, segment=1)


def test_io_nu_import_integ_missing(test_data_path: Path):
    path = test_data_path.joinpath("nu/normal")
    _, signals, _, _, _ = read_directory(path, cycle=1, segment=1)
    assert signals.shape == (40, 127)
    assert np.all(~np.isnan(signals[:29]))
    assert np.all(np.isnan(signals[29]))
    assert np.all(~np.isnan(signals[30:]))


def test_io_nu_import_autoblank(test_data_path: Path):
    path = test_data_path.joinpath("nu/autoblank")

    masses, signals, times, _, regions = read_directory(
        path, cycle=1, segment=1, autoblank="regions"
    )
    mass_idx = np.searchsorted(masses, (55.5, 89.5))

    assert np.allclose(regions[0]["mass"], (31.3255, 36.9595))  # sulphur or O2
    assert np.searchsorted(masses, regions[1]["mass"][0]) < mass_idx[1]
    assert np.searchsorted(masses, regions[1]["mass"][1]) >= mass_idx[1]
    assert np.allclose(regions[1]["time"], (2.1665, 2.6150))

    idx = np.searchsorted(times, regions[1]["time"])
    assert idx[0] == 20712
    assert idx[1] == 24999

    assert np.all(np.isnan(signals[20712:24999, mass_idx[1]]))
    assert np.all(~np.isnan(signals[20712:24999, mass_idx[0]]))

    _, signals, _, _, _ = read_directory(path, cycle=1, segment=1, autoblank="off")
    assert np.all(~np.isnan(signals[20712:24999, mass_idx[1]]))
    assert np.all(~np.isnan(signals[20712:24999, mass_idx[0]]))

    _, signals, _, _, _ = read_directory(path, cycle=1, segment=1, autoblank="all")
    assert np.all(np.isnan(signals[20712:24999, mass_idx[1]]))
    assert np.all(np.isnan(signals[20712:24999, mass_idx[0]]))


def test_select_nu_signals():
    masses = np.array([1.0, 2.0, 10.0])
    signals = np.reshape(np.arange(30), (3, 10)).T

    data = select_nu_signals(masses, signals, {"a": 2.0, "b": 9.95})
    assert data.dtype.names == ("a", "b")
    assert np.all(data["a"] == np.arange(10, 20))
    assert np.all(data["b"] == np.arange(20, 30))

    with pytest.raises(ValueError):
        select_nu_signals(masses, signals, {"a": 2.0, "b": 9.95}, max_mass_diff=0.01)


def test_nu_read_integ_binary_memmap(test_data_path: Path):
    path = test_data_path.joinpath("nu/normal/1.integ")

    data = read_integ_binary(path)
    data_memmap = read_integ_binary(path, memmap=True)

    assert np.all(data == data_memmap)
