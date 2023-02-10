from pathlib import Path

import numpy as np
import pytest

from spcal.io.nu import is_nu_directory, read_nu_directory


def test_is_nu_dir():
    path = Path(__file__).parent.joinpath("data/nu")
    assert is_nu_directory(path)
    assert not is_nu_directory(path.parent)
    assert not is_nu_directory(path.joinpath("fake"))

def test_io_nu_import():
    path = Path(__file__).parent.joinpath("data/nu")
    masses, signals, info = read_nu_directory(path)
    assert masses.size == 194
    assert np.isclose(masses[0][0], 22.98582197)
    assert np.isclose(masses[0][-1], 240.02343301)
    # with pytest.warns():
    #     data, old_names = import_single_particle_file(
    #         path,
    #         first_line=2,
    #         columns=(0, 1, 2),
    #         new_names=("a", "b", "c"),
    #         convert_cps=10.0,
    #     )
    # assert old_names == ["A", "B", "C"]
    # assert np.all(data["a"] == 10)
    # assert np.all(data["b"] == [10, 20, 30])
    # assert np.all(data["c"] == [10, 20, 40])


# def test_io_text_import_euro():
#     path = Path(__file__).parent.joinpath("data/text_euro.csv")
#     data, old_names = import_single_particle_file(
#         path, delimiter=";", first_line=2, columns=(0, 1, 2), new_names=("a", "b", "c")
#     )
#     assert old_names == ["A", "B", "C"]
#     assert np.all(data["a"] == 1)
#     assert np.all(data["b"] == [1, 2, 3])
#     assert np.all(data["c"] == [1, 2, 4])
