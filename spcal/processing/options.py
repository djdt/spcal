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

    def __repr__(self) -> str:  # pragma: no cover
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
        else:  # pragma: no cover
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

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SPCalIsotopeOptions(density={self.density}, response={self.response}, "
            f"mass_fraction={self.mass_fraction})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SPCalIsotopeOptions):  # pragma: no cover
            return False

        for attr in [
            "density",
            "response",
            "mass_fraction",
            "concentration",
            "diameter",
            "mass_response",
        ]:
            a = getattr(self, attr)
            b = getattr(other, attr)
            if a is None and getattr(other, attr) is not None or a != b:
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
        else:  # pragma: no cover
            raise ValueError(f"unknown calibration mode '{mode}'")

        if key == "mass":
            return mass_ok
        elif key in ["size", "volume"]:
            return mass_ok and self.density is not None and self.density > 0.0
        else:  # pragma: no cover
            raise ValueError(f"unknown calibration key '{key}'")


class SPCalLimitOptions(object):
    def __init__(
        self,
        limit_method: str = "automatic",
        gaussian_kws: dict | None = None,
        poisson_kws: dict | None = None,
        compound_poisson_kws: dict | None = None,
        window_size: int = 0,
        max_iterations: int = 1,
        single_ion_parameters: np.ndarray | None = None,
    ):
        self.limit_method = limit_method

        # deafult kws
        _gaussian_kws = {"alpha": 2.867e-7}
        _poisson_kws = {"alpha": 1e-3, "function": "formula c"}
        _compound_poisson_kws = {"alpha": 1e-6, "sigma": 0.5}

        if gaussian_kws is not None:  # pragma: no cover
            _gaussian_kws.update(gaussian_kws)
        if poisson_kws is not None:  # pragma: no cover
            _poisson_kws.update(poisson_kws)
        if compound_poisson_kws is not None:  # pragma: no cover
            _compound_poisson_kws.update(compound_poisson_kws)

        self.gaussian_kws = _gaussian_kws
        self.poisson_kws = _poisson_kws
        self.compound_poisson_kws = _compound_poisson_kws

        self.window_size = window_size
        self.max_iterations = max_iterations

        self.single_ion_parameters = single_ion_parameters
        self.manual_limits: dict[SPCalIsotopeBase, float] = {}
        self.default_manual_limit = 100.0

    def limitsForIsotope(
        self,
        data_file: SPCalDataFile,
        isotope: SPCalIsotopeBase,
        exclusion_regions: list[tuple[float, float]] | None = None,
        limit_method: str | None = None,
    ) -> SPCalLimit:
        signals = data_file[isotope]
        if exclusion_regions is not None and len(exclusion_regions) > 0:
            idx = np.searchsorted(data_file.times, exclusion_regions)
            signals = signals.copy()
            for start, end in idx:
                signals[start:end] = np.nan

        if limit_method is None:  # pragma: no cover
            limit_method = self.limit_method

        if limit_method == "manual input":
            if isotope not in self.manual_limits:
                logger.warning(f"no manual limit for isotope '{isotope}', using default")
            threshold = self.manual_limits.get(isotope, self.default_manual_limit)
            return SPCalLimit("Manual", float(np.nanmean(signals)), threshold)

        if limit_method == "automatic":
            if SPCalGaussianLimit.isGaussianDistributed(signals):
                limit_method = "gaussian"
            elif data_file.isTOF():
                limit_method = "compound poisson"
            else:
                limit_method = "poisson"

        if limit_method == "compound poisson" or (
            limit_method == "highest" and data_file.isTOF()
        ):
            # Override the default sigma if single ion paramters are present
            if self.single_ion_parameters is not None:
                if isinstance(isotope, SPCalIsotope):
                    if isotope.mass <= 0.0:  # pragma: no cover
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
                    if any(x <= 0.0 for x in masses):  # pragma: no cover
                        raise ValueError("isotope mass is 0")
                    sigma = np.mean(
                        np.interp(
                            masses,
                            self.single_ion_parameters["mass"],
                            self.single_ion_parameters["sigma"],
                        )
                    )
                else:  # pragma: no cover
                    raise ValueError(
                        f"cannot infer sigma from isotope type '{type(isotope)}'"
                    )
            else:
                sigma = self.compound_poisson_kws["sigma"]

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
            if not data_file.isTOF():  # pragma: no cover
                logger.warning("Compound-Poisson limit created for non-TOF data file")

            return SPCalCompoundPoissonLimit(
                signals,
                alpha=self.compound_poisson_kws["alpha"],
                sigma=sigma,  # type: ignore , sigma bound when method==compound poisson
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
        else:  # pragma: no cover
            raise ValueError(f"unknown limit method {limit_method}")
