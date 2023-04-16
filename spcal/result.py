"""Misc and helper calculation functions."""
import logging
from pathlib import Path
from typing import Dict

import numpy as np

from spcal import particle
from spcal.limit import SPCalLimit

logger = logging.getLogger(__name__)


class SPCalResult(object):
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
        self.indicies = np.flatnonzero(detections)

        self.background = np.nanmean(responses[labels == 0])
        self.background_error = np.nanstd(responses[labels == 0])

        self.limits = limits

        self.inputs = {}
        if inputs_kws is not None:
            self.inputs.update({k: v for k, v in inputs_kws.items() if v is not None})

    @property
    def events(self) -> int:
        return np.count_nonzero(~np.isnan(self.responses))

    @property
    def ionic_background(self) -> float | None:
        if "response" not in self.inputs:
            return None
        return self.background / self.inputs["response"]

    @property
    def number(self) -> int:
        return self.indicies.size

    @property
    def number_error(self) -> int:
        return np.sqrt(self.number)

    @property
    def mass_concentration(self) -> float | None:
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

    def asCellConcentration(
        self, value: float | np.ndarray
    ) -> float | np.ndarray | None:
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
        mass = self.asMass(value)
        if mass is not None and "density" in self.inputs:
            return particle.particle_size(mass, density=self.inputs["density"])
        return None

    def convertTo(
        self, value: float | np.ndarray, key: str
    ) -> float | np.ndarray | None:
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
