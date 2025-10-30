import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from spcal import particle
from spcal.datafile import SPCalDataFile
from spcal.detection import accumulate_detections
from spcal.limit import (
    SPCalCompoundPoissonLimit,
    SPCalGaussianLimit,
    SPCalLimit,
    SPCalPoissonLimit,
)

logger = logging.getLogger(__name__)


def calculate_result(
    method: "SPCalProcessingMethod",
    data_file: SPCalDataFile,
    isotope: str,
    max_size: int | None,
):
    limit = method.limit_options.limitsForIsotope(data_file, isotope)

    if method.accumulation_method == "signal mean":
        limit_accumulation = limit.mean_signal
    elif method.accumulation_method == "half detection threshold":
        limit_accumulation = (limit.mean_signal + limit.detection_threshold) / 2.0
    elif method.accumulation_method == "detection threshold":
        limit_accumulation = limit.detection_threshold
    else:
        raise ValueError(f"unknown accumulation method {method.accumulation_method}")

    limit_detection = limit.detection_threshold
    limit_accumulation = np.minimum(limit_accumulation, limit_detection)

    signals = data_file[isotope][:max_size]
    times = data_file.times[:max_size]

    detections, labels, regions = accumulate_detections(
        signals,
        limit_accumulation=limit_accumulation,
        limit_detection=limit_detection,
        points_required=method.points_required,
        prominence_required=method.prominence_required,
    )

    return SPCalProcessingResult(
        isotope,
        limit=limit,
        method=method,
        signals=signals,
        times=times,
        detections=detections,
        labels=labels,
        regions=regions,
    )


class SPCalInstrumentOptions(object):
    def __init__(
        self,
        event_time: float | None,
        uptake: float | None,
        efficiency: float | None,
    ):
        self.event_time = event_time
        self.uptake = uptake
        self.efficiency = efficiency

    def canCalibrate(self, key: str, mode: str = "efficiency") -> bool:
        if key == "signal":
            return True

        if mode == "efficiency":
            return all(
                x is not None and x > 0.0
                for x in [self.event_time, self.uptake, self.efficiency]
            )
        elif mode == "mass response":
            return True
        else:
            raise ValueError(f"unknown calibration mode '{mode}'")


class SPCalIsotopeOptions(object):
    def __init__(
        self,
        density: float | None,
        response: float | None,
        mass_fraction: float | None,
        concentration: float | None = None,
        diameter: float | None = None,
        mass_response: float | None = None,
    ):
        # used for general calibration
        self.density = density
        self.response = response
        self.mass_fraction = mass_fraction
        # used for reference
        self.concentration = concentration
        self.diameter = diameter
        # used for calibration via mass response
        self.mass_response = mass_response

    def canCalibrate(self, key: str, mode: str = "efficiency") -> bool:
        if key == "signal":
            return True

        if mode == "efficiency":
            mass_ok = all(
                x is not None and x > 0.0 for x in [self.response, self.mass_fraction]
            )
        elif mode == "mass response":
            mass_ok = all(
                x is not None and x > 0.0
                for x in [self.mass_response, self.mass_fraction]
            )
        else:
            raise ValueError(f"unknown calibration mode '{mode}'")

        if key == "mass":
            return mass_ok
        elif key in ["size", "volume"]:
            return mass_ok and self.density is not None and self.density > 0.0
        else:
            raise ValueError(f"unknown calibration key '{key}'")


class SPCalLimitOptions(object):
    def __init__(
        self,
        method: str = "automatic",
        gaussian_kws: dict | None = None,
        poisson_kws: dict | None = None,
        compound_poisson_kws: dict | None = None,
        window_size: int = 0,
        max_iterations: int = 1,
        single_ion_parameters: np.ndarray | None = None,
    ):
        self.method = method

        # deafult kws
        _gaussian_kws = {"alpha": 2.867e-7}
        _poisson_kws = {"alpha": 1e-3, "function": "formula c"}
        _compound_poisson_kws = {"alpha": 1e-6, "sigma": 0.5}

        if gaussian_kws is not None:
            _gaussian_kws.update(gaussian_kws)
        if poisson_kws is not None:
            _poisson_kws.update(poisson_kws)
        if compound_poisson_kws is not None:
            _compound_poisson_kws.update(compound_poisson_kws)

        self.gaussian_kws = _gaussian_kws
        self.poisson_kws = _poisson_kws
        self.compound_poisson_kws = _compound_poisson_kws

        self.window_size = window_size
        self.max_iterations = max_iterations

        self.single_ion_parameters = single_ion_parameters

    def limitsForIsotope(
        self, data_file: SPCalDataFile, isotope: str, method: str | None = None
    ) -> SPCalLimit:
        signals = data_file[isotope]
        if method is None:
            method = self.method

        if method == "automatic":
            if SPCalGaussianLimit.isGaussianDistributed(signals):
                method = "gaussian"
            elif data_file.isTOF():
                method = "compound poisson"
            else:
                method = "poisson"

        if method == "compound poisson" or (method == "highest" and data_file.isTOF()):
            # Override the default sigma if single ion paramters are present
            if self.single_ion_parameters is not None:
                sigma = np.interp(
                    data_file.isotopeMass(isotope),
                    self.single_ion_parameters["mass"],
                    self.single_ion_parameters["sigma"],
                )
            else:
                sigma = self.compound_poisson_kws["sigma"]

        if method == "gaussian":
            return SPCalGaussianLimit(
                signals,
                **self.gaussian_kws,
                window_size=self.window_size,
                max_iterations=self.max_iterations,
            )
        elif method == "poisson":
            if data_file.isTOF():
                logger.warning(
                    "Poisson limit created for TOF data file, use Compound-Poisson"
                )
            return SPCalPoissonLimit(
                signals,
                **self.poisson_kws,
                window_size=self.window_size,
                max_iterations=self.max_iterations,
            )
        elif method == "compound poisson":
            if not data_file.isTOF():
                logger.warning("Compound-Poisson limit created for non-TOF data file")

            return SPCalCompoundPoissonLimit(
                signals,
                alpha=self.compound_poisson_kws["alpha"],
                sigma=sigma,  # type: ignore , sigma bound when method==compound poisson
                window_size=self.window_size,
                max_iterations=self.max_iterations,
            )
        elif method == "highest":
            gaussian = SPCalGaussianLimit(
                signals,
                **self.gaussian_kws,
                window_size=self.window_size,
                max_iterations=self.max_iterations,
            )
            if data_file.isTOF():
                poisson = SPCalCompoundPoissonLimit(
                    signals,
                    alpha=self.compound_poisson_kws["alpha"],
                    sigma=sigma,  # type: ignore , sigma bound when method==highest and istof
                    window_size=self.window_size,
                    max_iterations=self.max_iterations,
                )
            else:
                poisson = SPCalPoissonLimit(
                    signals,
                    **self.poisson_kws,
                    window_size=self.window_size,
                    max_iterations=self.max_iterations,
                )
            if np.any(poisson.detection_threshold > gaussian.detection_threshold):
                return poisson
            else:
                return gaussian
        else:
            raise ValueError(f"unknown limit method {method}")


class SPCalProcessingResult(object):
    def __init__(
        self,
        isotope: str,
        limit: SPCalLimit,
        method: "SPCalProcessingMethod",
        signals: np.ndarray,
        times: np.ndarray,
        detections: np.ndarray,
        labels: np.ndarray,
        regions: np.ndarray,
        indicies: np.ndarray | None = None,
    ):
        self.isotope = isotope
        self.limit = limit
        self.method = method

        self.signals = signals
        self.times = times

        self.detections = detections
        self.labels = labels
        self.regions = regions

        self.peak_indicies = indicies
        self.filter_indicies = None

        self.background = float(np.nanmean(self.signals[self.labels == 0.0]))
        self.background_error = float(
            np.nanstd(self.signals[self.labels == 0.0], mean=self.background)
        )

        self._event_time: float | None = None  # cache

    @property
    def num_events(self) -> int:
        return self.signals.shape[0]

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
        masses = self.calibrated("mass")
        if masses is None:
            return None

        return particle.particle_total_concentration(
            masses,
            self.method.instrument_options.efficiency,  # type: ignore , checked via calibrate('mass')
            self.method.instrument_options.uptake,  # type: ignore , checked via calibrate('mass')
            self.total_time,
        )

    def calibrated(self, key: str) -> np.ndarray | None:
        if not self.method.canCalibrate(key, self.isotope):
            return None
        result = self.method.calibrateTo(self.detections, key, self.isotope)
        assert isinstance(result, np.ndarray)
        return result


class SPCalProcessingMethod(object):
    CALIBRATION_KEYS = ["signal", "mass", "size", "volume"]
    ACCUMULATION_METHODS = [
        "signal mean",
        "half detection threshold",
        "detection threshold",
    ]

    def __init__(
        self,
        instrument_options: SPCalInstrumentOptions,
        limit_options: SPCalLimitOptions,
        isotope_options: dict[str, SPCalIsotopeOptions],
        selected_isotopes: list[str],
        accumulation_method: str = "signal mean",
        points_required: int = 1,
        prominence_required: float = 0.2,
        calibration_mode: str = "efficiency",
    ):
        if accumulation_method not in SPCalProcessingMethod.ACCUMULATION_METHODS:
            raise ValueError(
                f"accumulation method must be one of {', '.join(SPCalProcessingMethod.ACCUMULATION_METHODS)}"
            )
        if calibration_mode not in ["efficiency", "mass response"]:
            raise ValueError(
                "calibration mode must be one of 'efficiency', 'mass response'"
            )

        self.instrument_options = instrument_options
        self.limit_options = limit_options
        self.isotope_options = isotope_options

        self.accumulation_method = accumulation_method
        self.points_required = points_required
        self.prominence_required = prominence_required

        self.calibration_mode = calibration_mode

    def processDataFile(
        self,
        data_file: SPCalDataFile,
        isotopes: list[str] | None = None,
        max_size: int | None = None,
    ) -> dict[str, "SPCalProcessingResult"]:
        results = {}
        if isotopes is None:
            isotopes = data_file.selected_isotopes

        with ThreadPoolExecutor() as exec:
            futures = [
                exec.submit(calculate_result, self, data_file, isotope, max_size)
                for isotope in isotopes
            ]
            results = {future.result().isotope: future.result() for future in futures}

        return results

    def canCalibrate(self, key: str, isotope: str) -> bool:
        if key not in SPCalProcessingMethod.CALIBRATION_KEYS:
            raise ValueError(f"unknown calibration key '{key}'")
        if isotope not in self.isotope_options:
            raise ValueError(f"unknown isotope '{isotope}'")

        return self.instrument_options.canCalibrate(
            key, self.calibration_mode
        ) and self.isotope_options[isotope].canCalibrate(key, self.calibration_mode)

    def calibrateTo(
        self, signals: float | np.ndarray, key: str, isotope: str
    ) -> float | np.ndarray:
        if key == "signal":
            return signals
        elif key == "mass":
            return self.calibrateToMass(signals, isotope)
        elif key == "size":
            return self.calibrateToSize(signals, isotope)
        elif key == "volume":
            return self.calibrateToVolume(signals, isotope)
        else:
            raise ValueError(f"unknown calibration key '{key}'")

    def calibrateToMass(
        self, signals: float | np.ndarray, isotope: str
    ) -> float | np.ndarray:
        mass_fraction = self.isotope_options[isotope].mass_fraction
        assert mass_fraction is not None
        if self.calibration_mode == "efficiency":
            assert self.instrument_options.event_time is not None
            assert self.instrument_options.uptake is not None
            assert self.instrument_options.efficiency is not None
            assert self.isotope_options[isotope].response is not None
            response = self.isotope_options[isotope].response
            assert response is not None

            return particle.particle_mass(
                signals,
                dwell=self.instrument_options.event_time,
                efficiency=self.instrument_options.efficiency,
                flow_rate=self.instrument_options.uptake,
                response_factor=response,
                mass_fraction=mass_fraction,
            )
        else:
            mass_response = self.isotope_options[isotope].mass_response
            assert mass_response is not None
            return signals * mass_response / mass_fraction

    def calibrateToSize(
        self, signals: float | np.ndarray, isotope: str
    ) -> float | np.ndarray:
        density = self.isotope_options[isotope].density
        assert density is not None
        return particle.particle_size(
            self.calibrateToMass(signals, isotope), density=density
        )

    def calibrateToVolume(
        self, signals: float | np.ndarray, isotope: str
    ) -> float | np.ndarray:
        density = self.isotope_options[isotope].density
        assert density is not None
        return self.calibrateToMass(signals, isotope) * density
