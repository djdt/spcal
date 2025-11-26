from concurrent.futures import ThreadPoolExecutor
import typing

import numpy as np

from spcal import particle
from spcal.cluster import agglomerative_cluster, prepare_data_for_clustering
from spcal.datafile import SPCalDataFile
from spcal.detection import accumulate_detections, combine_regions
from spcal.isotope import SPCalIsotope


from spcal.processing.options import (
    SPCalInstrumentOptions,
    SPCalIsotopeOptions,
    SPCalLimitOptions,
)
from spcal.processing.result import SPCalProcessingResult

if typing.TYPE_CHECKING:
    from spcal.processing.filter import SPCalProcessingFilter


class SPCalProcessingMethod(object):
    CALIBRATION_KEYS = ["signal", "mass", "size", "volume"]
    ACCUMULATION_METHODS = [
        "signal mean",
        "half detection threshold",
        "detection threshold",
    ]

    def __init__(
        self,
        instrument_options: SPCalInstrumentOptions | None = None,
        limit_options: SPCalLimitOptions | None = None,
        isotope_options: dict[SPCalIsotope, SPCalIsotopeOptions] | None = None,
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
        if instrument_options is None:
            instrument_options = SPCalInstrumentOptions(None, None, None)
        if limit_options is None:
            limit_options = SPCalLimitOptions()
        if isotope_options is None:
            isotope_options = {}

        self.instrument_options = instrument_options
        self.limit_options = limit_options
        self.isotope_options = isotope_options

        self.accumulation_method = accumulation_method
        self.points_required = points_required
        self.prominence_required = prominence_required

        self.calibration_mode = calibration_mode

        self.cluster_distance = cluster_distance

        self.filters: list[list["SPCalProcessingFilter"]] = [[]]
        self.exclusion_regions: list[tuple[float, float]] = []

    def setFilters(self, filters: list[list["SPCalProcessingFilter"]]):
        self.filters = filters

    @staticmethod
    def calculate_result_for_isotope(
        method: "SPCalProcessingMethod",
        data_file: SPCalDataFile,
        isotope: SPCalIsotope,
        max_size: int | None,
    ) -> SPCalProcessingResult:
        limit = method.limit_options.limitsForIsotope(
            data_file, isotope, method.exclusion_regions
        )

        if method.accumulation_method == "signal mean":
            limit_accumulation = limit.mean_signal
        elif method.accumulation_method == "half detection threshold":
            limit_accumulation = (limit.mean_signal + limit.detection_threshold) / 2.0
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
        # remove exclusion regions
        if len(method.exclusion_regions) > 0:
            idx = np.searchsorted(data_file.times, method.exclusion_regions)
            signals = signals.copy()
            for start, end in idx:
                signals[start:end] = np.nan

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

    def processDataFile(
        self,
        data_file: SPCalDataFile,
        isotopes: list[SPCalIsotope] | None = None,
        max_size: int | None = None,
    ) -> dict[SPCalIsotope, SPCalProcessingResult]:
        results = {}
        if isotopes is None:
            isotopes = data_file.selected_isotopes

        with ThreadPoolExecutor() as exec:
            futures = [
                exec.submit(
                    SPCalProcessingMethod.calculate_result_for_isotope,
                    self,
                    data_file,
                    isotope,
                    max_size,
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
                    "cannot cluster, peak_indicies have not been generated"
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
