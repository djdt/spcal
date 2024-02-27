"""Class for calculating results."""
import logging
from pathlib import Path

import numpy as np

from spcal import particle
from spcal.limit import SPCalLimit

logger = logging.getLogger(__name__)


class Filter(object):
    operations: dict[str, np.ufunc] = {
        ">": np.greater,
        "<": np.less,
        ">=": np.greater_equal,
        "<=": np.less_equal,
        "==": np.equal,
    }

    def __init__(self, name: str, unit: str, operation: str, value: float):
        self.name = name
        self.unit = unit
        self.operation = operation
        self.value = value

    def __repr__(self) -> str:
        return f"Filter({self.name}::{self.unit} {self.operation!r} {self.value!r})"

    @property
    def ufunc(self) -> np.ufunc:
        return Filter.operations[self.operation]

    def filter(self, results: dict[str, "SPCalResult"]) -> np.ndarray | None:
        if self.name not in results:
            return None
        data = results[self.name].convertTo(results[self.name].detections, self.unit)
        if data is None:
            return None
        return self.ufunc(data, self.value)

    @staticmethod
    def filter_results(
        filters: list[list["Filter"]], results: dict[str, "SPCalResult"]
    ) -> np.ndarray:
        """Filter a dictionary of results.

        Filters are stored as a list of groups where filters  within groups
        are combined by && (logical and) and each group is combined by || (logical or).

        Args:
            filters: list of filter groups
            results: dict of name: result

        Returns:
            indicies of filtered detections
        """
        size = next(iter(results.values())).detections.size
        valid = np.zeros(size, dtype=bool)

        for filter_group in filters:
            group_valid = np.ones(size, dtype=bool)
            for filter in filter_group:
                filter_valid = filter.filter(results)
                if filter_valid is not None:
                    group_valid = np.logical_and(group_valid, filter_valid)
            valid = np.logical_or(group_valid, valid)

        return np.flatnonzero(valid)


class ClusterFilter(object):
    def __init__(self, idx: int, unit: str):
        """idx is the index of the group in decsending order by size.
        i.e., 0=largest group"""
        self.idx = idx
        self.unit = unit

    def __repr__(self) -> str:
        return f"ClusterFilter({self.idx}::{self.unit})"

    def filter(self, cluster_results: dict[str, np.ndarray]) -> np.ndarray | None:
        if self.unit not in cluster_results:  # pragma: no cover
            return None
        counts = np.bincount(cluster_results[self.unit])
        idx = np.argsort(counts)[::-1]
        return cluster_results[self.unit] == idx[self.idx]

    @staticmethod
    def filter_clusters(
        filters: list["ClusterFilter"], clusters: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Filter a cluster indicies.

        key of cluster dict is from SPCalResult types.

        Args:
            filters: list of cluster filters
            results: dict of key: indicies

        Returns:
            indicies of filtered clusters
        """
        size = next(iter(clusters.values())).size
        valid = np.zeros(size, dtype=bool)

        for filter in filters:
            idx = filter.filter(clusters)
            if idx is not None:
                valid = np.logical_or(valid, idx)

        return np.flatnonzero(valid)


class SPCalResult(object):
    """Calculates results from single particle detections.

    At minimum `detections` must contain 'signal' key, other
    valid keys are 'mass', 'size', 'volume', 'cell_concentration'.

    Attributes:
        file: path of file results are from
        responses: structured array of single particle data
        detections: dict of type: particle detections array
        indicies: indices of non-zero detections
        background: mean of non-detection regions
        background_error: error of ``background``
        limits: SPCalLimit for element
        inputs: inputs used to calculate results
    """

    base_units = {
        "signal": ("counts", 1.0),
        "mass": ("kg", 1.0),
        "size": ("m", 1.0),
        "volume": ("m³", 1.0),
        "cell_concentration": ("mol/L", 1.0),
    }

    def __init__(
        self,
        file: str | Path,
        responses: np.ndarray,
        detections: np.ndarray,
        labels: np.ndarray,
        limits: SPCalLimit,
        inputs_kws: dict[str, float] | None = None,
    ):
        if detections.size == 0:
            raise ValueError("SPCalResult: detections size is zero")
        self.file = Path(file)

        self.responses = responses
        self.detections = detections
        self.indicies = np.flatnonzero(
            np.logical_and(detections > 0, np.isfinite(detections))
        )

        self.background = np.nanmean(responses[labels == 0])
        self.background_error = np.nanstd(responses[labels == 0])

        self.limits = limits

        self.inputs = {}
        if inputs_kws is not None:
            self.inputs.update({k: v for k, v in inputs_kws.items() if v is not None})

    @property
    def events(self) -> int:
        """Number of valid (non nan) events."""
        return np.count_nonzero(~np.isnan(self.responses))

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

    @property
    def mass_concentration(self) -> float | None:
        """Total particle concentration in kg/L.

        Retuires 'mass' type detections. 'efficiency', 'uptake' and 'time' inupts.

        Returns:
            concentration or None if unable to calculate
        """
        if any(x not in self.inputs for x in ["efficiency", "uptake", "time"]):
            return None

        masses = self.asMass(self.detections)
        if masses is None:
            return None

        return particle.particle_total_concentration(
            masses,
            efficiency=self.inputs["efficiency"],
            flow_rate=self.inputs["uptake"],
            time=self.inputs["time"],
        )

    @property
    def number_concentration(self) -> float | None:
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

    def __repr__(self) -> str:  # pragma: no cover
        return f"SPCalResult({self.number})"

    def asCellConcentration(
        self, value: float | np.ndarray
    ) -> float | np.ndarray | None:
        """Convert a value to an intracellur concentration in mol/L.

        Requires 'dwelltime', 'efficiency', 'uptake', 'response', 'mass_fraction',
        'cell_diameter' and 'molar_mass' inputs.

        Args:
            value: single value or array

        Returns:
            value or None if unable to calculate
        """
        masses = self.asMass(value)
        if masses is not None and all(
            x in self.inputs for x in ["cell_diameter", "molar_mass"]
        ):
            return particle.cell_concentration(
                masses,
                diameter=self.inputs["cell_diameter"],
                molar_mass=self.inputs["molar_mass"],
            )
        return None

    def canCalibrateMass(self, mode: str = "either") -> bool:
        if mode in ["efficiency", "either"] and all(
            x in self.inputs
            for x in [
                "dwelltime",
                "efficiency",
                "uptake",
                "response",
                "mass_fraction",
            ]
        ):
            return True
        if mode in ["mass response", "either"] and all(
            x in self.inputs for x in ["mass_response", "mass_fraction"]
        ):
            return True
        return False

    def canCalibrateSize(self) -> bool:
        return self.canCalibrateMass() and "density" in self.inputs

    def canCalibrateCellConcentration(self) -> bool:
        return (
            self.canCalibrateMass()
            and "cell_concentration" in self.inputs
            and "molar_mass" in self.inputs
        )

    def asMass(
        self, value: float | np.ndarray, mode: str = "either"
    ) -> float | np.ndarray | None:
        """Convert value to mass in kg.

        'mass_response' and 'mass_fraction' inputs.

        For the 'efficiency' mode: requires 'dwelltime', 'efficiency', 'uptake',
        'response' and 'mass_fraction' inputs.
        For the 'mass response' mode: requires 'mass response' and 'mass fraction'.
        Mode 'either' will try 'efficiency' then 'mass response'.

        Args:
            value: single value or array
            mode: method to calculate mass, 'efficiency', 'mass response' or 'either'

        Returns:
            value or None if unable to calculate
        """
        if mode in ["efficiency", "either"] and self.canCalibrateMass(mode):
            return particle.particle_mass(
                value,
                dwell=self.inputs["dwelltime"],
                efficiency=self.inputs["efficiency"],
                flow_rate=self.inputs["uptake"],
                response_factor=self.inputs["response"],
                mass_fraction=self.inputs["mass_fraction"],
            )
        if mode in ["mass response", "either"] and self.canCalibrateMass(mode):
            return value * self.inputs["mass_response"] / self.inputs["mass_fraction"]
        return None

    def asSize(self, value: float | np.ndarray) -> float | np.ndarray | None:
        """Convert value to size in m.

        Requires the ``asMass`` and 'density' inputs.

        Args:
            value: single value or array

        Returns:
            value or None if unable to calculate
        """
        masses = self.asMass(value)
        if masses is not None and "density" in self.inputs:
            return particle.particle_size(masses, density=self.inputs["density"])
        return None

    def asVolume(self, value: float | np.ndarray) -> float | np.ndarray | None:
        """Convert value to size in m.

        Requires the ``asMass`` and 'density' inputs.

        Args:
            value: single value or array

        Returns:
            value or None if unable to calculate
        """
        mass = self.asMass(value)
        if mass is not None and "density" in self.inputs:
            return mass * self.inputs["density"]
        return None

    def asVolume(self, value: float | np.ndarray) -> float | np.ndarray | None:
        """Convert value to size in m.

        Requires the ``asMass`` and 'density' inputs.

        Args:
            value: single value or array

        Returns:
            value or None if unable to calculate
        """
        mass = self.asMass(value)
        if mass is not None and "density" in self.inputs:
            return mass * self.inputs["density"]
        return None

    def convertTo(
        self, value: float | np.ndarray, key: str
    ) -> float | np.ndarray | None:
        """Helper function to convert to mass, size or conc.

        Args:
            value: single value or array
            key: type of conversion {'single', 'mass', 'size', 'volume',
                                     'cell_concentration'}

        Returns:
            converted value or None if unable to calculate
        """
        if key == "signal":
            return value
        elif key == "mass":
            return self.asMass(value)
        elif key == "size":
            return self.asSize(value)
        elif key == "volume":
            return self.asVolume(value)
        elif key == "cell_concentration":
            return self.asCellConcentration(value)
        else:
            raise KeyError(f"convertTo: unknown key '{key}'.")

    @staticmethod
    def all_valid_indicies(results: list["SPCalResult"]) -> np.ndarray:
        """Return the indices where any of the results are valid."""
        size = results[0].detections.size
        valid = np.zeros(size, dtype=bool)
        for result in results:
            valid[result.indicies] = True
        return np.flatnonzero(valid)
