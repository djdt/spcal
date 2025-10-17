import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

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


class SPCalInstrumentOptions(object):
    def __init__(
        self,
        event_time: float,
        uptake: float,
        efficiency: float,
    ):
        self.event_time = event_time
        self.uptake = uptake
        self.efficiency = efficiency

    def readyToCalibrate(self, key: str, mode: str = "efficiency") -> bool:
        if key == "signal":
            return True

        if mode == "efficiency":
            return self.uptake is not None and self.efficiency is not None
        elif mode == "mass response":
            return True
        #     return self.mass_response is not None
        else:
            raise ValueError(f"unknown calibration mode '{mode}'")


class SPCalIsotopeOptions(object):
    def __init__(
        self, density, response, molar_mass, mass_fraction, concentration, diameter
    ):
        self.density = density
        self.response = response
        self.diameter = diameter
        self.molar_mass = molar_mass
        self.mass_fraction = mass_fraction
        self.concentration = concentration
        self.diameter = diameter

    def readyToCalibrate(self, key: str, mode: str = "efficiency") -> bool:
        if key == "signal":
            return True

        if mode == "efficiency":
            mass_ok = self.response is not None and self.mass_fraction is not None
        elif mode == "mass response":
            mass_ok = self.mass_fraction is not None
        else:
            raise ValueError(f"unknown calibration mode '{mode}'")

        if key == "mass":
            return mass_ok
        elif key in ["size", "volume"]:
            return mass_ok and self.density is not None
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
        _gaussian_kws = {"alpha": 2.867e-7, "windows_size": 0}
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

            # Override the default sigma if single ion paramters are present
            if self.single_ion_parameters is not None:
                sigma = np.interp(
                    data_file.isotopeMass(isotope),
                    self.single_ion_parameters["mass"],
                    self.single_ion_parameters["sigma"],
                )
            else:
                sigma = self.compound_poisson_kws["sigma"]

            return SPCalCompoundPoissonLimit(
                signals,
                alpha=self.compound_poisson_kws["alpha"],
                sigma=sigma,
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
            poisson = SPCalPoissonLimit(
                signals,
                **self.poisson_kws,
                window_size=self.window_size,
                max_iterations=self.max_iterations,
            )
            if poisson.detection_threshold > gaussian.detection_threshold:
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
        instrument_options: SPCalInstrumentOptions,
        isotope_options: SPCalIsotopeOptions,
        signals: np.ndarray,
        times: np.ndarray,
        detections: np.ndarray,
        labels: np.ndarray,
        regions: np.ndarray,
        indicies: np.ndarray | None = None,
    ):
        self.isotope = isotope
        self.limit = limit
        self.instrument_options = instrument_options
        self.isotope_ooptions = isotope_options

        self.signals = signals
        self.times = times

        self.detections = detections
        self.labels = labels
        self.regions = regions

        self.peak_indicies = indicies
        self.filter_indicies = None

        self.background = np.nanmean(self.signals[self.labels == 0.0])
        self.background_error = np.nanstd(
            self.signals[self.labels == 0.0], mean=self.background
        )

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

    # @property
    # def ionic_background(self) -> float | None:
    #     """Background in kg/L.
    #
    #     Requires 'response' input.
    #     """
    #     if "response" not in self.inputs:
    #         return None
    #     return self.background / self.inputs["response"]

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


class SPCalProcessingMethod(object):
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
        self.instrument_options = instrument_options
        self.limit_options = limit_options
        self.isotope_options = isotope_options

        self.accumulation_method = accumulation_method
        self.points_required = points_required
        self.prominence_required = prominence_required

        self.calibration_mode = calibration_mode

    def processDataFile(
        self, data_file: SPCalDataFile, isotopes: list[str], max_size: int | None = None
    ) -> dict[np.ndarray, SPCalProcessingResult]:
        def calculateResult(data_file, isotope, max_size):
            limit = self.limit_options.limitsForIsotope(data_file, isotope)

            if self.accumulation_method == "mean signal":
                limit_accumulation = limit.mean_signal
            elif self.accumulation_method == "half detection threshold":
                limit_accumulation = (
                    limit.mean_signal + limit.detection_threshold
                ) / 2.0
            elif self.accumulation_method == "detection threshold":
                limit_accumulation = limit.detection_threshold
            else:
                raise ValueError(
                    f"unknown accumulation method {self.accumulation_method}"
                )

            limit_detection = limit.detection_threshold
            limit_accumulation = np.minimum(limit_accumulation, limit_detection)

            signals = data_file[isotope][:max_size]
            times = data_file.times[:max_size]

            detections, labels, regions = accumulate_detections(
                signals,
                limit_accumulation=limit_accumulation,
                limit_detection=limit_detection,
                points_required=self.points_required,
                prominence_required=self.prominence_required,
            )

            return SPCalProcessingResult(
                isotope,
                limit,
                self.instrument_options,
                self.isotope_options[isotope],
                signals,
                times,
                detections,
                labels,
                regions,
            )

        results = {}
        with ProcessPoolExecutor() as exec:
            futures = {
                exec.submit(calculateResult, data_file, isotope, max_size): isotope
                for isotope in isotopes
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()

        return results

    def calibrateToMass(
        self, signals: float | np.ndarray, isotope: str
    ) -> float | np.ndarray:
        if self.calibration_mode == "efficiency":
            return particle.particle_mass(
                signals,
                dwell=self.instrument_options.event_time,
                efficiency=self.instrument_options.efficiency,
                flow_rate=self.instrument_options.uptake,
                response_factor=self.isotope_options[isotope].response,
                mass_fraction=self.isotope_options[isotope].mass_fraction,
            )
        else:
            return (
                signals
                * self.instrument_options.mass_response
                / self.isotope_options[isotope].mass_fraction
            )

    def calibrateToSize(
        self, signals: float | np.ndarray, isotope: str
    ) -> float | np.ndarray:
        return particle.particle_size(
            self.calibrateToMass(signals, isotope),
            density=self.isotope_options[isotope].density,
        )

    def calibrateToVolume(
        self, signals: float | np.ndarray, isotope: str
    ) -> float | np.ndarray:
        return (
            self.calibrateToMass(signals, isotope)
            * self.isotope_options[isotope].density
        )

    def canCalibrate(self, key: str, isotope: str) -> bool:
        return self.instrument_options.readyToCalibrate(
            key, self.calibration_mode
        ) and self.isotope_options[isotope].readyToCalibrate(key, self.calibration_mode)

    def ionicBackground(self, result: SPCalProcessingResult) -> float:
        return result.background / self.isotope_options[result.isotope].response

    def massConcentration(self, result: SPCalProcessingResult) -> float:
        return particle.particle_total_concentration(
            self.calibrateToMass(result.detections, result.isotope),
            self.instrument_options.efficiency,
            self.instrument_options.uptake,
            self.instrument_options.event_time * result.num_events,
        )

    def numberConcentration(self, result: SPCalProcessingResult) -> float:
        return np.around(
            particle.particle_number_concentration(
                result.number,
                self.instrument_options.efficiency,
                self.instrument_options.uptake,
                self.instrument_options.event_time * result.num_events,
            )
        )
