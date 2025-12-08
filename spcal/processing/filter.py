from typing import Callable

import numpy as np

from spcal.isotope import SPCalIsotopeBase

from spcal.processing.result import SPCalProcessingResult
from spcal.processing.method import SPCalProcessingMethod


class SPCalProcessingFilter(object):
    def __init__(self, isotope: SPCalIsotopeBase | None):
        self.isotope = isotope

    def invalidPeaks(self, result: SPCalProcessingResult) -> np.ndarray:
        raise NotImplementedError

    def validPeaks(self, result: SPCalProcessingResult) -> np.ndarray:
        raise NotImplementedError


class SPCalClusterFilter(SPCalProcessingFilter):
    pass


class SPCalValueFilter(SPCalProcessingFilter):
    def __init__(
        self,
        isotope: SPCalIsotopeBase,
        key: str,
        operation: Callable[[np.ndarray, float], np.ndarray],
        value: float,
    ):
        if key not in SPCalProcessingMethod.CALIBRATION_KEYS:
            raise ValueError(f"invalid key {key}")
        super().__init__(isotope)
        self.key = key
        self.operation = operation
        self.value = value

    def invalidPeaks(self, result: SPCalProcessingResult) -> np.ndarray:
        if result.peak_indicies is None:
            raise ValueError("peak indicies have not been calculated")
        if not result.canCalibrate(self.key):
            return result.peak_indicies
        return result.peak_indicies[
            np.logical_not(
                self.operation(result.calibrated(self.key, filtered=False), self.value)
            )
        ]

    def validPeaks(self, result: SPCalProcessingResult) -> np.ndarray:
        if result.peak_indicies is None:
            raise ValueError("peak indicies have not been calculated")
        if not result.canCalibrate(self.key):
            return np.array([])
        return result.peak_indicies[
            self.operation(result.calibrated(self.key, filtered=False), self.value)
        ]


class SPCalTimeFilter(SPCalProcessingFilter):
    def __init__(self, start: float, end: float):
        super().__init__(None)
        self.start, self.end = start, end

    def invalidPeaks(self, result: SPCalProcessingResult) -> np.ndarray:
        if result.peak_indicies is None:
            raise ValueError("peak indicies have not been calculated")
        peak_times = result.times[result.maxima]
        return result.peak_indicies[
            np.logical_and(peak_times >= self.start, peak_times <= self.end)
        ]

    def validPeaks(self, result: SPCalProcessingResult) -> np.ndarray:
        if result.peak_indicies is None:
            raise ValueError("peak indicies have not been calculated")
        peak_times = result.times[result.maxima]
        return result.peak_indicies[
            np.logical_or(peak_times < self.start, peak_times > self.end)
        ]
