import numpy as np

from spcal import particle
from spcal.detection import background_mask, detection_maxima
from spcal.isotope import SPCalIsotope
from spcal.limit import SPCalLimit

import typing

if typing.TYPE_CHECKING:
    from spcal.processing.method import SPCalProcessingMethod


class SPCalProcessingResult(object):
    def __init__(
        self,
        isotope: SPCalIsotope,
        limit: SPCalLimit,
        method: "SPCalProcessingMethod",
        signals: np.ndarray,
        times: np.ndarray,
        detections: np.ndarray,
        regions: np.ndarray,
        indicies: np.ndarray | None = None,
    ):
        self.isotope = isotope
        self.limit = limit
        self.method = method

        self.signals = signals
        self.times = times

        self.detections = detections
        self.maxima = detection_maxima(signals, regions)
        self.regions = regions

        self.peak_indicies = indicies
        self.filter_indicies = np.arange(detections.size)

        mask = background_mask(regions, signals.size)

        self.background = float(np.nanmean(signals[mask]))
        self.background_error = float(np.nanstd(signals[mask], mean=self.background))

        self._event_time: float | None = None  # cache

    def __repr__(self) -> str:
        return f"SPCalProcessingResult(number={self.number})"

    @property
    def num_events(self) -> int:
        return int(np.count_nonzero(np.any(np.isscalar(self.signals), axis=1)))

    @property
    def event_time(self) -> float:
        if self._event_time is None:
            self._event_time = float(np.mean(np.diff(self.times)))
        return self._event_time

    @property
    def total_time(self) -> float:
        return self.event_time * self.num_events

    @property
    def valid_events(self) -> int:
        """Number of valid (non nan) events."""
        return np.count_nonzero(~np.isnan(self.signals))  # type: ignore , numpy int is fine

    @property
    def ionic_background(self) -> float | None:
        """Background in kg/L.

        Requires 'response' input.
        """
        response = self.method.isotope_options[self.isotope].response
        if response is None:
            return None
        return float(self.background / response)

    @property
    def number(self) -> int:
        """Number of non-zero detections."""
        if self.filter_indicies is None:
            return self.detections.size
        else:
            return self.filter_indicies.size

    @property
    def number_error(self) -> int:
        """Sqrt of ``number``."""
        return np.sqrt(self.number)

    @property
    def number_concentration(self) -> float | None:
        if not self.method.instrument_options.canCalibrate("mass", "efficiency"):
            return None
        else:
            return np.around(
                particle.particle_number_concentration(
                    self.number,
                    self.method.instrument_options.efficiency,  # type: ignore , checked via canCalibrate('mass', 'efficiency')
                    self.method.instrument_options.uptake,  # type: ignore , checked via canCalibrate('mass', 'efficiency')
                    self.event_time,
                )
            )

    @property
    def mass_concentration(self) -> float | None:
        if not self.canCalibrate("mass"):
            return None
        masses = self.calibrated("mass")

        return particle.particle_total_concentration(
            masses,
            self.method.instrument_options.efficiency,  # type: ignore , checked via calibrate('mass')
            self.method.instrument_options.uptake,  # type: ignore , checked via calibrate('mass')
            self.total_time,
        )

    def canCalibrate(self, key: str) -> bool:
        return self.method.canCalibrate(key, self.isotope)

    def calibrated(self, key: str, filtered: bool = True) -> np.ndarray:
        values = self.detections
        if filtered:
            values = values[self.filter_indicies]
        result = self.method.calibrateTo(values, key, self.isotope)
        assert isinstance(result, np.ndarray)
        return result
