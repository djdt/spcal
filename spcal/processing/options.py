import logging

import numpy as np

from spcal.datafile import SPCalDataFile
from spcal.isotope import SPCalIsotope, SPCalIsotopeBase, SPCalIsotopeExpression
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
        uptake: float | None,
        efficiency: float | None,
    ):
        self.uptake = uptake
        self.efficiency = efficiency

    def __repr__(self) -> str:
        return f"SPCalInstrumentOptions(uptake={self.uptake}, efficiency={self.efficiency})"

    def canCalibrate(self, key: str, mode: str = "efficiency") -> bool:
        if key == "signal":
            return True

        if mode == "efficiency":
            return all(
                x is not None and x > 0.0 for x in [self.uptake, self.efficiency]
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SPCalIsotopeOptions):
            return False

        if not self.density != other.density:
            return False
        if not self.response != other.response:
            return False
        if not self.mass_fraction != other.mass_fraction:
            return False
        if not self.concentration != other.concentration:
            return False
        if not self.diameter != other.diameter:
            return False
        if not self.mass_response != other.mass_response:
            return False

        return True

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
        self,
        data_file: SPCalDataFile,
        isotope: SPCalIsotopeBase,
        exclusion_regions: list[tuple[float, float]] | None = None,
        method: str | None = None,
    ) -> SPCalLimit:
        signals = data_file[isotope]
        if exclusion_regions is not None and len(exclusion_regions) > 0:
            idx = np.searchsorted(data_file.times, exclusion_regions)
            signals = signals.copy()
            for start, end in idx:
                signals[start:end] = np.nan

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
                if isinstance(isotope, SPCalIsotope):
                    if isotope.mass <= 0.0:
                        raise ValueError("isotope mass is 0")
                    sigma = np.interp(
                        isotope.mass,
                        self.single_ion_parameters["mass"],
                        self.single_ion_parameters["sigma"],
                    )
                elif isinstance(isotope, SPCalIsotopeExpression):
                    masses = [
                        token.mass
                        for token in isotope.tokens
                        if isinstance(token, SPCalIsotope)
                    ]
                    if any(x <= 0.0 for x in masses):
                        raise ValueError("isotope mass is 0")
                    sigma = np.mean(
                        np.interp(
                            masses,
                            self.single_ion_parameters["mass"],
                            self.single_ion_parameters["sigma"],
                        )
                    )
                else:
                    raise ValueError(
                        f"cannot infer sigma from isotope type '{type(isotope)}'"
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
