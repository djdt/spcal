from typing import Callable
from spcal.datafile import SPCalTextDataFile, SPCalDataFile
from pathlib import Path
import numpy as np
import random
import pytest

from spcal.isotope import ISOTOPE_TABLE, SPCalIsotope
from spcal.limit import SPCalLimit
from spcal.processing.method import SPCalProcessingMethod
from spcal.processing.result import SPCalProcessingResult


def pytest_sessionstart(session):
    np.seterr(invalid="raise", divide="raise")


@pytest.fixture(scope="function")
def default_method() -> SPCalProcessingMethod:
    method = SPCalProcessingMethod()
    method.limit_options.poisson_kws["function"] = "currie"
    method.limit_options.poisson_kws["alpha"] = 1e-3
    method.limit_options.compound_poisson_kws["alpha"] = 1e-6
    method.limit_options.compound_poisson_kws["sigma"] = 0.65
    method.processing_options.prominence_required = 0.2
    return method


@pytest.fixture(scope="session")
def test_data_path() -> Path:
    return Path(__file__).parent.joinpath("data")


# @pytest.fixture(scope="module")
# def random_result_generator() -> Callable[..., SPCalProcessingResult]:
#     def random_result(
#         method: SPCalProcessingMethod,
#         isotope: SPCalIsotope | None = None,
#         size: int = 1000,
#         number: int = 10,
#     ) -> SPCalProcessingResult:
#         signals = np.random.poisson(1.0, size=size).astype(np.float32)
#         times = np.linspace(0.001, 1, size)
#
#         detections = np.random.uniform(10.0, 50.0, number)
#         regions = np.random.choice(size - 200, number)
#         signals[regions + 100] = detections
#         regions = np.stack((regions + 100, regions + 101), axis=1)
#
#         if isotope is None:
#             isotope = random.choice(list(ISOTOPE_TABLE.values()))
#
#         return SPCalProcessingResult(
#             isotope,
#             SPCalLimit("TestLimit", 1.0, 3.0),
#             method,
#             0.001,
#             Path("/home/fake_path.datafile"),
#             signals,
#             times,
#             detections,
#             regions,
#         )
#
#     return random_result


@pytest.fixture(scope="function")
def random_datafile_generator() -> Callable[..., SPCalDataFile]:
    def random_datafile(
        size: int = 1000,
        number: int | list[np.ndarray] = 10,
        lam: float = 1.0,
        isotopes: list[SPCalIsotope] | None = None,
        path: Path | None = None,
        seed: int | None = None,
    ):
        if seed is not None:
            np.random.seed(seed)

        if isotopes is None:
            isotopes = [ISOTOPE_TABLE[("Au", 197)]]
        data = np.empty(
            size, dtype=[(str(isotope), np.float32) for isotope in isotopes]
        )
        assert data.dtype.names is not None
        for i, name in enumerate(data.dtype.names):
            data[name] = np.random.poisson(lam=lam, size=size)
            if isinstance(number, int):
                pos = np.random.choice(size, number)
            else:
                pos = number[i]
            data[name][pos] += np.random.normal(lam * 50, size=pos.size)

        df = SPCalTextDataFile(
            path or Path("/fake/path/text.csv"),
            data,
            np.linspace(0.0, 1.0, size),
            isotope_table={isotope: str(isotope) for isotope in isotopes},
        )
        df.selected_isotopes = isotopes
        return df

    return random_datafile
