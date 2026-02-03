import zipfile
from pathlib import Path

import numpy as np
import pytest

from spcal.io.nu import (
    eventtime_from_info,
    is_nu_directory,
    is_nu_run_info_file,
    read_directory,
    select_nu_signals,
)


def test_is_nu_run_info_file(test_data_path: Path):
    path = test_data_path.joinpath("nu")
    assert not is_nu_run_info_file(path)
    assert is_nu_run_info_file(path.joinpath("run.info"))


def test_is_nu_dir(test_data_path: Path):
    path = test_data_path.joinpath("nu")
    assert is_nu_directory(path)
    assert not is_nu_directory(path.parent)
    assert not is_nu_directory(path.joinpath("fake"))


def test_io_nu_import(test_data_path: Path):
    path = test_data_path.joinpath("nu")
    masses, signals, times, info = read_directory(path, cycle=1, segment=1)
    assert masses.size == 127
    assert signals.shape == (40, 127)
    assert np.isclose(masses[0], 80.90475)
    assert np.isclose(masses[-1], 208.9560)

    assert np.all(np.isclose(info["MassCalCoefficients"], [-0.221135, 0.000613]))
    assert len(info["SegmentInfo"]) == 1
    assert info["SegmentInfo"][0]["AcquisitionTriggerDelayNs"] == 14950.0

    assert np.isclose(eventtime_from_info(info), 0.09824e-3)


def test_io_nu_import_integ_limits(test_data_path: Path):
    path = test_data_path.joinpath("nu")
    masses, signals, times, info = read_directory(
        path, last_integ_file=1, cycle=1, segment=1
    )
    assert masses.size == 127
    assert times.shape == (10,)
    assert signals.shape == (10, 127)

    masses, signals, times, info = read_directory(
        path, first_integ_file=1, last_integ_file=3, cycle=1, segment=1
    )
    assert masses.size == 127
    assert times.shape == (20,)
    assert signals.shape == (20, 127)

    with pytest.raises(ValueError):
        masses, signals, times, info = read_directory(
            path, first_integ_file=1, last_integ_file=1, cycle=1, segment=1
        )


def test_io_nu_import_integ_missing(test_data_path: Path):
    path = test_data_path.joinpath("nu")
    masses, signals, times, info = read_directory(path, cycle=1, segment=1)
    assert signals.shape == (40, 127)
    assert np.all(~np.isnan(signals[:29]))
    assert np.all(np.isnan(signals[29]))
    assert np.all(~np.isnan(signals[30:]))


def test_io_nu_import_autoblank(test_data_path: Path, tmp_path: Path):
    path = test_data_path.joinpath("nu/autob.zip")
    zp = zipfile.ZipFile(path)
    zp.extractall(tmp_path)

    masses, signals, times, info = read_directory(
        tmp_path.joinpath("autob"), cycle=1, segment=1
    )
    # blanking region of mass idx 10-14 at 32204 - 49999
    assert np.all(np.isnan(signals[32204:49999, 0:14]))
    assert np.all(~np.isnan(signals[32204:49999, 14:]))

    masses, signals, times, info = read_directory(
        tmp_path.joinpath("autob"), cycle=1, segment=1, autoblank=False
    )
    assert np.all(~np.isnan(signals[32204:49999, 0:14]))


def test_select_nu_signals():
    masses = np.array([1.0, 2.0, 10.0])
    signals = np.reshape(np.arange(30), (3, 10)).T

    data = select_nu_signals(masses, signals, {"a": 2.0, "b": 9.95})
    assert data.dtype.names == ("a", "b")
    assert np.all(data["a"] == np.arange(10, 20))
    assert np.all(data["b"] == np.arange(20, 30))

    with pytest.raises(ValueError):
        select_nu_signals(masses, signals, {"a": 2.0, "b": 9.95}, max_mass_diff=0.01)
