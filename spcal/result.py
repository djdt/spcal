"""Misc and helper calculation functions."""
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

import spcal
from spcal.limit import SPCalLimit


class SPCalResult(object):
    def __init__(
        self,
        file: str | Path,
        responses: np.ndarray,
        detections: np.ndarray,
        labels: np.ndarray,
        limits: SPCalLimit | None = None,
        inputs_kws: Dict[str, float] | None = None,
    ):
        self.file = Path(file)

        self.responses = responses
        self.detections = {"signal": detections}
        self.indicies = np.flatnonzero(detections)

        self.background = np.mean(responses[labels == 0])
        self.background_error = np.std(responses[labels == 0])

        self.limits = limits

        self.inputs = {}
        if inputs_kws is not None:
            self.inputs.update({k: v for k, v in inputs_kws.items() if v is not None})

    @property
    def events(self) -> int:
        return self.responses.size

    @property
    def ionic_background(self) -> float | None:
        if not "response" in self.inputs:
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
        return spcal.particle_total_concentration(
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
            spcal.particle_number_concentration(
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
            return spcal.cell_concentration(
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
            return spcal.particle_mass(
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
            return spcal.particle_size(mass, density=self.inputs["density"])
        return None

    def fromNebulisationEfficiency(
        self,
    ) -> None:
        if any(
            x not in self.inputs
            for x in ["dwelltime", "efficiency", "uptake", "response", "mass_fraction"]
        ):
            raise ValueError("fromNebulisationEfficiency: missing required mass input")

        self.detections["mass"] = np.asarray(
            spcal.particle_mass(
                self.detections["signal"],
                dwell=self.inputs["dwelltime"],
                efficiency=self.inputs["efficiency"],
                flow_rate=self.inputs["uptake"],
                response_factor=self.inputs["response"],
                mass_fraction=self.inputs["mass_fraction"],
            )
        )
        if "density" not in self.inputs:
            Warning("fromNebulisationEfficiency: missing required size input")
        else:
            self.detections["size"] = np.asarray(
                spcal.particle_size(
                    self.detections["mass"], density=self.inputs["density"]
                )
            )

        if all(x in self.inputs for x in ["cell_diameter", "molar_mass"]):
            self.detections["cell_concentration"] = np.asarray(
                spcal.cell_concentration(
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
        if "density" not in self.inputs:
            Warning("fromMassResponse: missing required size input")
        else:
            self.detections["size"] = np.asarray(
                spcal.particle_size(
                    self.detections["mass"], density=self.inputs["density"]
                )
            )

        if all(x in self.inputs for x in ["cell_diameter", "molar_mass"]):
            self.detections["cell_concentration"] = np.asarray(
                spcal.cell_concentration(
                    self.detections["mass"],
                    diameter=self.inputs["cell_diameter"],
                    molar_mass=self.inputs["molar_mass"],
                )
            )
