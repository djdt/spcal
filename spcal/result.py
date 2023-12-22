"""Class for calculating results."""
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

from spcal import particle
from spcal.limit import SPCalLimit

logger = logging.getLogger(__name__)


class Filter(object):
    operations: Dict[str, np.ufunc] = {
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

    def filter(self, results: Dict[str, "SPCalResult"]) -> np.ndarray | None:
        if self.name not in results or self.unit not in results[self.name].detections:
            return None
        return self.ufunc(results[self.name].detections[self.unit], self.value)


class ClusterFilter(object):
    def __init__(self, idx: int, unit: str):
        self.idx = idx
        self.unit = unit

    def filter(self, cluster_results: Dict[str, np.ndarray]) -> np.ndarray | None:
        if self.unit not in cluster_results:
            return None
        counts = np.bincount(cluster_results[self.unit])
        idx = np.argsort(counts)[::-1]
        return cluster_results[self.unit] == idx[self.idx]


class SPCalResult(object):
    """Calculates results from single particle detections.

    At minimum `detections` must contain 'signal' key, other
    valid keys are 'mass', 'size', 'cell_concentration'.

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

    def __init__(
        self,
        file: str | Path,
        responses: np.ndarray,
        detections: np.ndarray,
        labels: np.ndarray,
        limits: SPCalLimit,
        inputs_kws: Dict[str, float] | None = None,
    ):
        if detections.size == 0:
            raise ValueError("SPCalResult: detections size is zero")
        self.file = Path(file)

        self.responses = responses
        self.detections = {"signal": detections}
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
        if "mass" not in self.detections or any(
            x not in self.inputs for x in ["efficiency", "uptake", "time"]
        ):
            return None
        return particle.particle_total_concentration(
            self.detections["mass"],
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

    def __repr__(self) -> str:
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
        mass = self.asMass(value)
        if mass is not None and all(
            x in self.inputs for x in ["cell_diameter", "molar_mass"]
        ):
            return particle.cell_concentration(
                mass,
                diameter=self.inputs["cell_diameter"],
                molar_mass=self.inputs["molar_mass"],
            )
        return None

    def asMass(self, value: float | np.ndarray) -> float | np.ndarray | None:
        """Convert value to mass in kg.

        Requires 'dwelltime', 'efficiency', 'uptake', 'response' and 'mass_fraction' or
        'mass_response' and 'mass_fraction' inputs.

        Args:
            value: single value or array

        Returns:
            value or None if unable to calculate
        """
        if all(  # Via efficiency
            x in self.inputs
            for x in ["dwelltime", "efficiency", "uptake", "response", "mass_fraction"]
        ):
            return particle.particle_mass(
                value,
                dwell=self.inputs["dwelltime"],
                efficiency=self.inputs["efficiency"],
                flow_rate=self.inputs["uptake"],
                response_factor=self.inputs["response"],
                mass_fraction=self.inputs["mass_fraction"],
            )
        elif all(x in self.inputs for x in ["mass_response", "mass_fraction"]):
            # Via mass response
            return value * self.inputs["mass_response"] / self.inputs["mass_fraction"]
        else:
            return None

    def asSize(self, value: float | np.ndarray) -> float | np.ndarray | None:
        """Convert value to mass in kg.

        Requires 'dwelltime', 'density', 'efficiency', 'uptake', 'response' and
        'mass_fraction' or 'density', 'mass_response' and 'mass_fraction' inputs.

        Args:
            value: single value or array

        Returns:
            value or None if unable to calculate
        """
        mass = self.asMass(value)
        if mass is not None and "density" in self.inputs:
            return particle.particle_size(mass, density=self.inputs["density"])
        return None

    def convertTo(
        self, value: float | np.ndarray, key: str
    ) -> float | np.ndarray | None:
        """Helper function to convert to mass, size or conc.

        Args:
            value: single value or array
            key: type of conversion {'single', 'mass', 'size', 'cell_concentration'}

        Returns:
            converted value or None if unable to calculate
        """
        if key == "signal":
            return value
        elif key == "mass":
            return self.asMass(value)
        elif key == "size":
            return self.asSize(value)
        elif key == "cell_concentration":
            return self.asCellConcentration(value)
        else:
            raise KeyError(f"convertTo: unknown key '{key}'.")

    def fromNebulisationEfficiency(
        self,
    ) -> None:
        """Calculates detection mass, size and intracellular concentration.

        Performs calibration of detections into masses.
        Requires the 'dwelltime', 'efficiency', 'uptake', 'response' and 'mass_fraction'
        inputs.

        Particle sizes are calculated if 'density' is available and intracellular
        concentrations if the 'cell_diameter' and 'molar_mass' inputs are available.
        """
        if any(
            x not in self.inputs
            for x in ["dwelltime", "efficiency", "uptake", "response", "mass_fraction"]
        ):
            raise ValueError("fromNebulisationEfficiency: missing required mass input")

        self.detections["mass"] = np.asarray(
            particle.particle_mass(
                self.detections["signal"],
                dwell=self.inputs["dwelltime"],
                efficiency=self.inputs["efficiency"],
                flow_rate=self.inputs["uptake"],
                response_factor=self.inputs["response"],
                mass_fraction=self.inputs["mass_fraction"],
            )
        )
        if "density" not in self.inputs:  # pragma: no cover
            logger.warning("fromNebulisationEfficiency: missing required size input")
        else:
            self.detections["size"] = np.asarray(
                particle.particle_size(
                    self.detections["mass"], density=self.inputs["density"]
                )
            )

        if all(x in self.inputs for x in ["cell_diameter", "molar_mass"]):
            self.detections["cell_concentration"] = np.asarray(
                particle.cell_concentration(
                    self.detections["mass"],
                    diameter=self.inputs["cell_diameter"],
                    molar_mass=self.inputs["molar_mass"],
                )
            )

    def fromMassResponse(self) -> None:
        """Calculates detection mass, size and intracellular concentration.

        Performs calibration of detections into masses.
        Requires the 'mass_response' and 'mass_fraction' inputs.

        Particle sizes are calculated if 'density' is available and intracellular
        concentrations if the 'cell_diameter' and 'molar_mass' inputs are available.
        """
        if any(x not in self.inputs for x in ["mass_response", "mass_fraction"]):
            raise ValueError("fromMassResponse: missing required mass input")

        self.detections["mass"] = self.detections["signal"] * (
            self.inputs["mass_response"] / self.inputs["mass_fraction"]
        )
        if "density" not in self.inputs:  # pragma: no cover
            logger.warning("fromMassResponse: missing required size input")
        else:
            self.detections["size"] = np.asarray(
                particle.particle_size(
                    self.detections["mass"], density=self.inputs["density"]
                )
            )

        if all(x in self.inputs for x in ["cell_diameter", "molar_mass"]):
            self.detections["cell_concentration"] = np.asarray(
                particle.cell_concentration(
                    self.detections["mass"],
                    diameter=self.inputs["cell_diameter"],
                    molar_mass=self.inputs["molar_mass"],
                )
            )


def filter_results(
    filters: List[List[Filter]], results: Dict[str, SPCalResult]
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
    size = next(iter(results.values())).detections["signal"].size
    valid = np.zeros(size, dtype=bool)

    for filter_group in filters:
        group_valid = np.ones(size, dtype=bool)
        for filter in filter_group:
            filter_valid = filter.filter(results)
            if filter_valid is not None:
                group_valid = np.logical_and(group_valid, filter_valid)
        valid = np.logical_or(group_valid, valid)

    return np.flatnonzero(valid)
