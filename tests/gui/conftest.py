from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore
from spcal.datafile import SPCalTextDataFile
from spcal.isotope import ISOTOPE_TABLE, SPCalIsotope


@pytest.fixture(
    scope="module",
    autouse=True,
    params=[QtCore.QLocale.Language.C],
)
def test_locales(request):
    locale = QtCore.QLocale(request.param)
    locale.setNumberOptions(
        locale.NumberOption.OmitGroupSeparator
        | locale.NumberOption.RejectGroupSeparator
    )
    QtCore.QLocale.setDefault(locale)


@pytest.fixture(scope="function")
def random_datafile_gen():
    def random_datafile(
        size: int = 100,
        number: int = 10,
        lam: float = 1.0,
        isotopes: list[SPCalIsotope] | None = None,
        path: Path | None = None,
        seed: int | None = None,
    ):
        if seed is not None:
            np.random.seed(seed)

        if isotopes is None:
            isotopes = [ISOTOPE_TABLE[("Ag", 109)], ISOTOPE_TABLE[("Au", 197)]]
        data = np.empty(
            size, dtype=[(str(isotope), np.float32) for isotope in isotopes]
        )
        assert data.dtype.names is not None
        for name in data.dtype.names:
            data[name] = np.random.poisson(lam=lam, size=size)
            data[name][np.random.choice(size, number)] += np.random.normal(
                lam * 20, size=number
            )

        df = SPCalTextDataFile(
            path or Path(),
            data,
            np.linspace(0.0, 1.0, size),
            isotope_table={isotope: str(isotope) for isotope in isotopes},
        )
        df.selected_isotopes = isotopes
        return df

    return random_datafile
