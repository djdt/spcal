from pathlib import Path

import numpy as np
import pytest

from spcal.io.text import read_single_particle_file


def test_io_text_import():
    path = Path(__file__).parent.joinpath("data/text/text_normal.csv")
    with pytest.warns():
        data, old_names = read_single_particle_file(
            path,
            first_line=2,
            columns=(0, 1, 2),
            new_names=("a", "b", "c"),
            convert_cps=10.0,
        )
    assert old_names == ["A", "B", "C"]
    assert np.all(data["a"] == 10)
    assert np.all(data["b"] == [10, 20, 30])
    assert np.all(data["c"] == [10, 20, 40])


def test_io_text_import_euro():
    path = Path(__file__).parent.joinpath("data/text/text_euro.csv")
    data, old_names = read_single_particle_file(
        path, delimiter=";", first_line=2, columns=(0, 1, 2), new_names=("a", "b", "c")
    )
    assert old_names == ["A", "B", "C"]
    assert np.all(data["a"] == 1)
    assert np.all(data["b"] == [1, 2, 3])
    assert np.all(data["c"] == [1, 2, 4])
