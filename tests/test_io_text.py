from pathlib import Path

import numpy as np
import pytest

from spcal.io.text import is_text_file, read_single_particle_file


def test_io_is_text_file():
    assert is_text_file(Path(__file__).parent.joinpath("data/text/text_normal.csv"))
    assert not is_text_file(Path(__file__).parent.joinpath("data/text/text_normal.bad"))
    assert not is_text_file(Path(__file__).parent.joinpath("data/text/"))


def test_io_text_import():
    path = Path(__file__).parent.joinpath("data/text/text_normal.csv")
    with pytest.warns():
        data = read_single_particle_file(
            path,
            first_line=2,
            columns=(0, 1, 2),
            convert_cps=10.0,
        )
    assert np.all(data["A"] == 10)
    assert np.all(data["B"] == [10, 20, 30])
    assert np.all(data["C"] == [10, 20, 40])


def test_io_text_import_euro():
    path = Path(__file__).parent.joinpath("data/text/text_euro.csv")
    data = read_single_particle_file(
        path, delimiter=";", first_line=2, columns=(0, 1, 2)
    )
    assert np.all(data["A"] == 1)
    assert np.all(data["B"] == [1, 2, 3])
    assert np.all(data["C"] == [1, 2, 4])
