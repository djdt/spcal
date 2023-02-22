from pathlib import Path

import numpy as np

from spcal.io.tofwerk import (
    calibrate_index_to_mass,
    calibrate_mass_to_index,
    is_tofwerk_file,
)


# def test_is_tofwerk_file():
#     path = Path(__file__).parent.joinpath("data/nu")


def test_calibration():
    mass = np.load(Path(__file__).parent.joinpath("data/tofwerk/massaxis.npy"))
    idx = np.arange(mass.size)

    mode = 2
    ps = [2827.69757786, -5221.04079879, 0.49997113]

    mass_to_idx = np.around(calibrate_mass_to_index(mass, mode, ps), 0).astype(int)
    assert np.all(mass_to_idx == idx)

    idx_to_mass = calibrate_index_to_mass(idx, mode, ps)
    assert np.allclose(idx_to_mass, mass)
