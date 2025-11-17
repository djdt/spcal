import numpy as np


def pytest_sessionstart(session):
    np.seterr(invalid="raise")
