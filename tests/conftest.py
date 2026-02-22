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
    method.limit_options.compound_poisson_kws["sigma"] = 0.65
    return method


@pytest.fixture(scope="session")
def test_data_path() -> Path:
    return Path(__file__).parent.joinpath("data")


@pytest.fixture(scope="module")
def random_result_generator():
    def random_result(
        method: SPCalProcessingMethod,
        isotope: SPCalIsotope | None = None,
        size: int = 1000,
        number: int = 10,
    ) -> SPCalProcessingResult:
        signals = np.random.poisson(1.0, size=size).astype(np.float32)
        times = np.linspace(0.001, 1, size)

        detections = np.random.uniform(10.0, 50.0, number)
        regions = np.random.choice(size - 200, number)
        signals[regions + 100] = detections
        regions = np.stack((regions + 100, regions + 101), axis=1)

        if isotope is None:
            isotope = random.choice(list(ISOTOPE_TABLE.values()))

        return SPCalProcessingResult(
            isotope,
            SPCalLimit("TestLimit", 1.0, 3.0),
            method,
            0.001,
            signals,
            times,
            detections,
            regions,
        )

    return random_result
