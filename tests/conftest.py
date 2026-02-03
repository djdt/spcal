from pathlib import Path
import numpy as np
import random
import pytest

from spcal.isotope import ISOTOPE_TABLE, SPCalIsotope
from spcal.limit import SPCalLimit
from spcal.processing.method import SPCalProcessingMethod
from spcal.processing.result import SPCalProcessingResult


def pytest_sessionstart(session):
    np.seterr(invalid="raise")


@pytest.fixture(scope="module")
def default_method() -> SPCalProcessingMethod:
    method = SPCalProcessingMethod()
    return method


@pytest.fixture(scope="session")
def test_data_path() -> Path:
    return Path(__file__).parent.joinpath("data")

@pytest.fixture(scope="module")
def random_result_generator():
    def random_result(
        method: SPCalProcessingMethod, isotope: SPCalIsotope | None = None
    ) -> SPCalProcessingResult:
        signals = np.random.poisson(1.0, size=1000)
        times = np.linspace(0.001, 1, 1000)

        detections = np.random.uniform(10.0, 50.0, 10)
        regions = np.random.choice(800, 10)
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
