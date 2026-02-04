from typing import Callable

import numpy as np

from spcal.isotope import SPCalIsotopeBase

from spcal.processing import CALIBRATION_KEYS
from spcal.processing.result import SPCalProcessingResult


class SPCalResultFilter(object):
    """A filter that takes an SPCalProcessingResult."""

    def preferInvalid(self) -> bool:  # pragma: no cover
        return False

    def invalidPeaks(
        self, result: SPCalProcessingResult
    ) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def validPeaks(
        self, result: SPCalProcessingResult
    ) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class SPCalIndexFilter(object):
    """A filter that compares existing indicies."""

    def __init__(self, operation: Callable[[np.ndarray, int], np.ndarray], index: int):
        self.operation = operation
        self.index = index

    def preferInvalid(self) -> bool:  # pragma: no cover
        return False

    def invalidPeaks(self, indicies: np.ndarray) -> np.ndarray:
        return np.flatnonzero(np.logical_not(self.operation(indicies, self.index)))

    def validPeaks(self, indicies: np.ndarray) -> np.ndarray:
        return np.flatnonzero(self.operation(indicies, self.index))


class SPCalClusterFilter(SPCalIndexFilter):
    def __init__(self, key: str, index: int):
        super().__init__(np.equal, index)
        self.key = key


class SPCalValueFilter(SPCalResultFilter):
    def __init__(
        self,
        isotope: SPCalIsotopeBase,
        key: str,
        operation: Callable[[np.ndarray, float], np.ndarray],
        value: float,
        prefer_invalid: bool = False,
    ):
        if key not in CALIBRATION_KEYS:  # pragma: no cover
            raise ValueError(f"invalid key {key}")
        super().__init__()
        self.isotope = isotope
        self.key = key
        self.operation = operation
        self.value = value

        self.prefer_invalid = prefer_invalid

    def preferInvalid(self) -> bool:
        return self.prefer_invalid

    def invalidPeaks(self, result: SPCalProcessingResult) -> np.ndarray:
        if result.peak_indicies is None:  # pragma: no cover
            raise ValueError("peak indicies have not been calculated")
        if not result.canCalibrate(self.key):  # pragma: no cover
            return result.peak_indicies
        return result.peak_indicies[
            np.logical_not(
                self.operation(result.calibrated(self.key, filtered=False), self.value)
            )
        ]

    def validPeaks(self, result: SPCalProcessingResult) -> np.ndarray:
        if result.peak_indicies is None:  # pragma: no cover
            raise ValueError("peak indicies have not been calculated")
        if not result.canCalibrate(self.key):  # pragma: no cover
            return np.array([])
        return result.peak_indicies[
            self.operation(result.calibrated(self.key, filtered=False), self.value)
        ]


class SPCalTimeFilter(SPCalResultFilter):
    def __init__(self, start: float, end: float):
        super().__init__()
        self.start, self.end = start, end

    def invalidPeaks(self, result: SPCalProcessingResult) -> np.ndarray:
        if result.peak_indicies is None:  # pragma: no cover
            raise ValueError("peak indicies have not been calculated")
        peak_times = result.times[result.maxima]
        return result.peak_indicies[
            np.logical_and(peak_times >= self.start, peak_times <= self.end)
        ]

    def validPeaks(
        self, result: SPCalProcessingResult
    ) -> np.ndarray:  # pragma: no cover , unused
        if result.peak_indicies is None:  # pragma: no cover
            raise ValueError("peak indicies have not been calculated")
        peak_times = result.times[result.maxima]
        return result.peak_indicies[
            np.logical_or(peak_times < self.start, peak_times > self.end)
        ]
