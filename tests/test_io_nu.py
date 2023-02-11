from pathlib import Path

import numpy as np
import pytest

from spcal.io.nu import (  # get_masses_from_nu_data,
    is_nu_directory,
    read_nu_directory,
    select_nu_signals,
)


def test_is_nu_dir():
    path = Path(__file__).parent.joinpath("data/nu")
    assert is_nu_directory(path)
    assert not is_nu_directory(path.parent)
    assert not is_nu_directory(path.joinpath("fake"))


def test_io_nu_import():
    path = Path(__file__).parent.joinpath("data/nu")
    masses, signals, info = read_nu_directory(path)
    assert masses.size == 194
    assert signals.shape == (30, 194)
    assert np.isclose(masses[0], 22.98582197)
    assert np.isclose(masses[-1], 240.02343301)

    assert np.all(
        np.isclose(info["MassCalCoefficients"], [-0.21539236835, 6.13507083932e-4])
    )
    assert len(info["SegmentInfo"]) == 1
    assert info["SegmentInfo"][0]["AcquisitionTriggerDelayNs"] == 8000.0

    # fmt: off
    assert np.all(  # sodium
        np.isclose(
            signals[:, 0],
            [
                3.2, 3.3, 1.1, 5.6, 2.0, 1.5, 0.0, 2.7, 3.9, 0.9,
                1.5, 0.0, 1.4, 1.4, 5.1, 2.2, 0.6, 0.9, 5.5, 4.3,
                1.2, 5.2, 1.6, 3.6, 2.8, 7.4, 1.1, 6.7, 2.3, 2.7
            ],
            atol=0.05,
        )
    )
    # fmt: on
    assert np.all(signals[:, 193] == 0.0)  # uranium?


def test_select_nu_signals():
    masses = np.array([1.0, 2.0, 10.0])
    signals = np.reshape(np.arange(30), (3, 10)).T

    data = select_nu_signals(masses, signals, {"a": 2.0, "b": 9.95})
    assert data.dtype.names == ("a", "b")
    assert np.all(data["a"] == np.arange(10, 20))
    assert np.all(data["b"] == np.arange(20, 30))

    with pytest.raises(ValueError):
        select_nu_signals(masses, signals, {"a": 2.0, "b": 9.95}, max_mass_diff=0.01)
