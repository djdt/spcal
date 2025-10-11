from typing import Any
from spcal.limit import (
    SPCalLimit,
    SPCalGaussianLimit,
    SPCalPoissonLimit,
    SPCalCompoundPoissonLimit,
)

from spcal.datafile import SPCalDataFile

from spcal.detection import accumulate_detections

import numpy as np

import logging

logger = logging.getLogger(__name__)


class SPCalProcessingResult(object):
    def __init__(
        self,
        signals: np.ndarray,
        limit: SPCalLimit,
        detections: np.ndarray,
        labels: np.ndarray,
        regions: np.ndarray,
        indicies: np.ndarray | None = None,
    ):
        self.signals = signals

        self.limit = limit

        self.detections = detections
        self.labels = labels
        self.regions = regions

        self.indicies = indicies

        self.background = np.nanmean(self.signals[self.labels == 0.0])
        self.background_error = np.nanstd(
            self.signals[self.labels == 0.0], mean=self.background
        )

    @property
    def num_events(self) -> int:
        return self.signals.shape[0]

    @property
    def valid_events(self) -> int:
        """Number of valid (non nan) events."""
        return np.count_nonzero(~np.isnan(self.signals))  # type: ignore , numpy int is fine

    @property
    def ionic_background(self) -> float | None:
        """Background in kg/L.

        Requires 'response' input.
        """
        if "response" not in self.inputs:
            return None
        return self.background / self.inputs["response"]

    @property
    def number(self) -> int:
        """Number of non-zero detections."""
        return self.indicies.size

    @property
    def number_error(self) -> int:
        """Sqrt of ``number``."""
        return np.sqrt(self.number)


class SPCalInstrumentOptions(object):
    def __init__(self, uptake, efficiency, efficiency_method):
        pass

class SPCalIsotopeOptions(object):
    def __init__(self, density, response, molar_mass, mass_fraction):
        pass


class SPCalProcessingMethod(object):
    def __init__(
        self,
        selected_isotopes: list[str],
        limit_method: str = "automatic",
        limit_accumulation_method: str = "signal mean",
        gaussian_kws: dict[str, Any] | None = None,
        poisson_kws: dict[str, Any] | None = None,
        compound_poisson_kws: dict[str, Any] | None = None,
        window_size: int = 0,
        max_iterations: int = 1,
        points_required: int = 1,
        prominence_required: float = 0.2,
        single_ion_parameters: np.ndarray | None = None,
    ):
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

        self.isotopes = selected_isotopes

        self.limit_method = limit_method
        self.window_size = window_size
        self.max_iterations = max_iterations

        self.gaussian_kws = _gaussian_kws
        self.poisson_kws = _poisson_kws
        self.compound_poisson_kws = _compound_poisson_kws

        self.accumulation_method = limit_accumulation_method
        self.points_required = points_required
        self.prominence_required = prominence_required

        self.single_ion_parameters = None

    def limitsForIsotope(self, data_file: SPCalDataFile, isotope: str) -> SPCalLimit:
        signals = data_file[isotope]
        limit_method = self.limit_method

        if limit_method == "automatic":
            if SPCalGaussianLimit.isGaussianDistributed(signals):
                limit_method = "gaussian"
            elif data_file.isTOF():
                limit_method = "compound poisson"
            else:
                limit_method = "poisson"

        if limit_method == "gaussian":
            return SPCalGaussianLimit(
                signals,
                **self.gaussian_kws,
                window_size=self.window_size,
                max_iterations=self.max_iterations,
            )
        elif limit_method == "poisson":
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
        elif limit_method == "compound poisson":
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
        elif limit_method == "highest":
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
            raise ValueError(f"unknown limit method {limit_method}")

    def processDataFile(
        self, data_file: SPCalDataFile
    ) -> dict[np.ndarray, SPCalProcessingResult]:
        results = {}

        for isotope in self.isotopes:
            limit = self.limitsForIsotope(data_file, isotope)

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

            detections, labels, regions = accumulate_detections(
                data_file[isotope],
                limit_accumulation=limit_accumulation,
                limit_detection=limit_detection,
                points_required=self.points_required,
                prominence_required=self.prominence_required,
            )

            result = SPCalProcessingResult(
                data_file[isotope], limit, detections, labels, regions
            )
            results[isotope] = result

        return results

    def massConcentration(self, result: SPCalProcessingResult) -> float | None:
        """Total particle concentration in kg/L.

        Requires 'mass' type detections. 'efficiency', 'uptake' and 'time' inupts.

        Returns:
            concentration or None if unable to calculate
        """
        if not self.canCalibrate("mass") or any(
            x not in self.inputs for x in ["efficiency", "uptake", "time"]
        ):
            return None

        return particle.particle_total_concentration(
            self.asMass(result.detections[result.indicies]),
            efficiency=self.options.efficiency,
            flow_rate=self.inputs["uptake"],
            time=self.inputs["time"],
        )

    def numberConcentration(self, result: SPCalProcessingResult) -> float | None:
        """Particle number concentration in #/L.

        Requires 'efficiency', 'uptake' and 'time' inputs.

        Returns:
            concentration or None if unable to calculate
        """
        if any(x not in self.inputs for x in ["efficiency", "uptake", "time"]):
            return None
        return np.around(
            particle.particle_number_concentration(
                self.number,
                efficiency=self.inputs["efficiency"],
                flow_rate=self.inputs["uptake"],
                time=self.inputs["time"],
            )
        )

    # def asCellConcentration(self, value: float | np.ndarray) -> float | np.ndarray:
    #     """Convert a value to an intracellur concentration in mol/L.
    #
    #     Requires 'dwelltime', 'efficiency', 'uptake', 'response', 'mass_fraction',
    #     'cell_diameter' and 'molar_mass' inputs.
    #
    #     Args:
    #         value: single value or array
    #
    #     Returns:
    #         value
    #     """
    #     return particle.cell_concentration(
    #         self.asMass(value),
    #         diameter=self.inputs["cell_diameter"],
    #         molar_mass=self.inputs["molar_mass"],
    #     )

    def asMass(self, value: float | np.ndarray) -> float | np.ndarray:
        """Convert value to mass in kg.

        'mass_response' and 'mass_fraction' inputs.

        For ``calibration_mode`` 'efficiency' mode: requires 'dwelltime', 'efficiency',
        'uptake', 'response' and 'mass_fraction' inputs.
        For ``calibration_mode`` 'mass response': requires 'mass_response' and
        'mass_fraction'.

        Args:
            value: single value or array

        Returns:
            value
        """
        if self.calibration_mode == "efficiency":
            return particle.particle_mass(
                value,
                dwell=self.inputs["dwelltime"],
                efficiency=self.inputs["efficiency"],
                flow_rate=self.inputs["uptake"],
                response_factor=self.inputs["response"],
                mass_fraction=self.inputs["mass_fraction"],
            )
        else:
            return value * self.inputs["mass_response"] / self.inputs["mass_fraction"]

    def asSize(self, value: float | np.ndarray) -> float | np.ndarray:
        """Convert value to size in m.

        Requires the ``asMass`` and 'density' inputs.

        Args:
            value: single value or array

        Returns:
            value
        """
        return particle.particle_size(
            self.asMass(value), density=self.inputs["density"]
        )

    def asVolume(self, value: float | np.ndarray) -> float | np.ndarray:
        """Convert value to size in m3.

        Requires the ``asMass`` and 'density' inputs.

        Args:
            value: single value or array

        Returns:
            value
        """
        return self.asMass(value) * self.inputs["density"]

    def calibrated(self, key: str, use_indicies: bool = True) -> np.ndarray:
        """Return calibrated detections.

        Also caches calibrated results.

        Args:
            key: key of ``base_units``
            all: only return values at ``self.indicies``

        Returns:
            array of calibrated detections
        """
        if key not in self._cache:
            self._cache[key] = np.asanyarray(self.convertTo(self.detections, key))
        if use_indicies:
            return self._cache[key][self.indicies]
        else:
            return self._cache[key]

    def canCalibrate(self, key: str) -> bool:
        if key == "signal":
            return True

        # all non signal conversions require mass
        if self.calibration_mode == "efficiency":
            mass = all(
                x in self.inputs
                for x in [
                    "dwelltime",
                    "efficiency",
                    "uptake",
                    "response",
                    "mass_fraction",
                ]
            )
        else:
            mass = all(x in self.inputs for x in ["mass_response", "mass_fraction"])

        if key == "mass":
            return mass
        elif key in ["size", "volume"]:
            return mass and "density" in self.inputs
        elif key == "cell_concentration":
            return (
                mass and "cell_diameter" in self.inputs and "molar_mass" in self.inputs
            )
        else:  # pragma: no cover
            raise KeyError(f"unknown key '{key}'.")

    def convertTo(self, value: float | np.ndarray, key: str) -> float | np.ndarray:
        """Helper function to convert to mass, size or conc.

        Args:
            value: single value or array
            key: type of conversion {'single', 'mass', 'size', 'volume',
                                     'cell_concentration'}

        Returns:
            converted value
        """
        if key == "signal":
            return value
        elif key == "cell_concentration":
            return self.asCellConcentration(value)
        elif key == "mass":
            return self.asMass(value)
        elif key == "size":
            return self.asSize(value)
        elif key == "volume":
            return self.asVolume(value)
        else:
            raise KeyError(f"convertTo: unknown key '{key}'.")

    @staticmethod
    def all_valid_indicies(results: list["SPCalResult"]) -> np.ndarray:
        """Return the indices where any of the results are valid."""
        if len(results) == 0:
            return np.array([], dtype=int)
        size = results[0].detections.size
        valid = np.zeros(size, dtype=bool)
        for result in results:
            valid[result.indicies] = True
        return np.flatnonzero(valid)
