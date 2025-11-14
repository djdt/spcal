import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np

from spcal import particle
from spcal.cluster import agglomerative_cluster, prepare_data_for_clustering
from spcal.datafile import SPCalDataFile
from spcal.detection import (
    accumulate_detections,
    background_mask,
    combine_regions,
    detection_maxima,
)
from spcal.isotope import SPCalIsotope
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

    def __repr__(self) -> str:
        return (
            f"SPCalIsotopeOptions(density={self.density}, response={self.response}, "
            f"mass_fraction={self.mass_fraction})"
        )

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
        self, data_file: SPCalDataFile, isotope: SPCalIsotope, method: str | None = None
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
                if isotope.mass <= 0.0:
                    logger.warning(f"invalid mass for {isotope}, {isotope.mass}")
                    sigma = self.compound_poisson_kws["sigma"]
                else:
                    sigma = np.interp(
                        isotope.mass,
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


class SPCalProcessingFilter(object):
    def __init__(self, isotope: SPCalIsotope | None):
        self.isotope = isotope

    def invalidPeaks(self, result: SPCalProcessingResult) -> np.ndarray:
        raise NotImplementedError

    def validPeaks(self, result: SPCalProcessingResult) -> np.ndarray:
        raise NotImplementedError


class SPCalValueFilter(SPCalProcessingFilter):
    def __init__(
        self,
        isotope: SPCalIsotope,
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
        isotope_options: dict[SPCalIsotope, SPCalIsotopeOptions],
        selected_isotopes: list[SPCalIsotope],
        accumulation_method: str = "signal mean",
        points_required: int = 1,
        prominence_required: float = 0.2,
        calibration_mode: str = "efficiency",
        cluster_distance: float = 0.03,
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

        self.cluster_distance = cluster_distance

        self.filters: list[list[SPCalProcessingFilter]] = [[]]

    def setFilters(self, filters: list[list[SPCalProcessingFilter]]):
        self.filters = filters

    def processDataFile(
        self,
        data_file: SPCalDataFile,
        isotopes: list[SPCalIsotope] | None = None,
        max_size: int | None = None,
    ) -> dict[SPCalIsotope, SPCalProcessingResult]:
        def calculate_result_for_isotope(
            method: "SPCalProcessingMethod",
            data_file: SPCalDataFile,
            isotope: SPCalIsotope,
            max_size: int | None,
        ) -> SPCalProcessingResult:
            limit = method.limit_options.limitsForIsotope(data_file, isotope)

            if method.accumulation_method == "signal mean":
                limit_accumulation = limit.mean_signal
            elif method.accumulation_method == "half detection threshold":
                limit_accumulation = (
                    limit.mean_signal + limit.detection_threshold
                ) / 2.0
            elif method.accumulation_method == "detection threshold":
                limit_accumulation = limit.detection_threshold
            else:
                raise ValueError(
                    f"unknown accumulation method {method.accumulation_method}"
                )

            limit_detection = limit.detection_threshold
            limit_accumulation = np.minimum(limit_accumulation, limit_detection)

            signals = data_file[isotope][:max_size]
            times = data_file.times[:max_size]

            detections, regions = accumulate_detections(
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
                regions=regions,
            )

        results = {}
        if isotopes is None:
            isotopes = data_file.selected_isotopes

        with ThreadPoolExecutor() as exec:
            futures = [
                exec.submit(
                    calculate_result_for_isotope, self, data_file, isotope, max_size
                )
                for isotope in isotopes
            ]
            results = {future.result().isotope: future.result() for future in futures}

        return results

    def filterResults(
        self, results: dict[SPCalIsotope, SPCalProcessingResult]
    ) -> dict[SPCalIsotope, SPCalProcessingResult]:
        # combined regions for multi-element peaks
        all_regions = combine_regions(
            [result.regions for result in results.values()], 2
        )
        for result in results.values():
            result.peak_indicies = (
                np.searchsorted(all_regions[:, 0], result.regions[:, 0], side="right")
                - 1
            )
        # filter results
        valid_peaks = []
        for filter_group in self.filters:
            group_valid = np.arange(all_regions.size)
            for filter in filter_group:
                if filter.isotope is None:  # isotope not important, e.g. time based
                    for result in results.values():
                        filter_invalid = filter.invalidPeaks(result)
                        group_valid = np.setdiff1d(
                            group_valid, filter_invalid, assume_unique=True
                        )
                elif filter.isotope in results:
                    filter_valid = filter.validPeaks(results[filter.isotope])
                    group_valid = np.intersect1d(
                        group_valid, filter_valid, assume_unique=True
                    )
            valid_peaks = np.union1d(group_valid, valid_peaks)

        for result in results.values():
            result.filter_indicies = np.flatnonzero(
                np.in1d(result.peak_indicies, valid_peaks)  # type: ignore , set above
            )
        return results

    def processClusters(
        self, results: dict[SPCalIsotope, SPCalProcessingResult], key: str = "signal"
    ) -> np.ndarray:
        npeaks = (
            np.amax(
                [
                    result.peak_indicies[-1]
                    for result in results.values()
                    if result.peak_indicies is not None
                ]
            )
            + 1
        )
        peak_data = np.zeros((npeaks, len(results)), np.float32)
        for i, result in enumerate(results.values()):
            if result.peak_indicies is None:
                raise ValueError(
                    "cannot cluster, peak_indicies have not been gerneated"
                )
            if not result.canCalibrate(key):
                continue
            np.add.at(
                peak_data[:, i],
                result.peak_indicies[result.filter_indicies],
                result.calibrated(key),
            )
        X = prepare_data_for_clustering(peak_data)
        return agglomerative_cluster(X, self.cluster_distance)

    def canCalibrate(self, key: str, isotope: SPCalIsotope) -> bool:
        if key not in SPCalProcessingMethod.CALIBRATION_KEYS:
            raise ValueError(f"unknown calibration key '{key}'")
        if isotope not in self.isotope_options:
            raise ValueError(f"unknown isotope '{isotope}'")

        return self.instrument_options.canCalibrate(
            key, self.calibration_mode
        ) and self.isotope_options[isotope].canCalibrate(key, self.calibration_mode)

    def calibrateTo(
        self, signals: float | np.ndarray, key: str, isotope: SPCalIsotope
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
        self, signals: float | np.ndarray, isotope: SPCalIsotope
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
        self, signals: float | np.ndarray, isotope: SPCalIsotope
    ) -> float | np.ndarray:
        density = self.isotope_options[isotope].density
        assert density is not None
        return particle.particle_size(
            self.calibrateToMass(signals, isotope), density=density
        )

    def calibrateToVolume(
        self, signals: float | np.ndarray, isotope: SPCalIsotope
    ) -> float | np.ndarray:
        density = self.isotope_options[isotope].density
        assert density is not None
        return self.calibrateToMass(signals, isotope) * density
