"""Misc and helper calculation functions."""
from bisect import bisect_left, insort
from typing import Dict, Tuple

import numpy as np

import spcal
from spcal.poisson import formula_c as poisson_limits

try:
    import bottleneck as bn

    BOTTLENECK_FOUND = True
except ImportError:  # pragma: no cover
    BOTTLENECK_FOUND = False


def moving_mean(x: np.ndarray, n: int) -> np.ndarray:
    """Calculates the rolling mean.

    Uses bottleneck.move_mean if available otherwise np.cumsum based algorithm.

    Args:
        x: array
        n: window size
    """
    if BOTTLENECK_FOUND:  # pragma: no cover
        return bn.move_mean(x, n)[n - 1 :]
    r = np.cumsum(x)
    r[n:] = r[n:] - r[:-n]  # type: ignore
    return r[n - 1 :] / n


def moving_median(x: np.ndarray, n: int) -> np.ndarray:
    """Calculates the rolling median.

    Uses bottleneck.move_median if available otherwise sort based algorithm.

    Args:
        x: array
        n: window size
    """
    if BOTTLENECK_FOUND:  # pragma: no cover
        return bn.move_median(x, n)[n - 1 :]

    r = np.empty(x.size - n + 1, x.dtype)
    sort = sorted(x[:n])
    m = n // 2
    m2 = m + n % 2 - 1

    for start in range(x.size - n):
        r[start] = sort[m] + sort[m2]
        end = start + n
        del sort[bisect_left(sort, x[start])]
        insort(sort, x[end])

    r[-1] = sort[m] + sort[m2]
    return r / 2.0


def moving_std(x: np.ndarray, n: int) -> np.ndarray:
    """Calculates the rolling standard deviation.

    Uses bottleneck.move_std if available otherwise np.cumsum based algorithm.

    Args:
        x: array
        n: window size
    """
    if BOTTLENECK_FOUND:  # pragma: no cover
        return bn.move_std(x, n)[n - 1 :]

    sums = np.empty(x.size - n + 1)
    sqrs = np.empty(x.size - n + 1)

    tab = np.cumsum(x) / n
    sums[0] = tab[n - 1]
    sums[1:] = tab[n:] - tab[:-n]

    tab = np.cumsum(x * x) / n
    sqrs[0] = tab[n - 1]
    sqrs[1:] = tab[n:] - tab[:-n]

    return np.sqrt(sqrs - sums * sums)


class SPCalLimit(object):
    def __init__(
        self,
        mean_background: float | np.ndarray,
        limit_of_criticality: float | np.ndarray,
        limit_of_detection: float | np.ndarray,
        name: str,
        params: Dict[str, float],
        window_size: int = 0,
    ):
        self.mean_background = mean_background
        self.limit_of_criticality = limit_of_criticality
        self.limit_of_detection = limit_of_detection

        self.name = name
        self.params = params
        self.window_size = window_size

    @classmethod
    def fromGaussian(
        cls,
        responses: np.ndarray,
        sigma: float = 3.0,
        window_size: int = 0,
        use_median: bool = False,
    ) -> "SPCalLimit":
        if responses.size == 0:
            raise ValueError("fromGaussian: responses is size 0")

        if window_size == 0:  # No window
            mean = np.median(responses) if use_median else np.mean(responses)
            std = np.std(responses)
        else:
            pad = np.pad(
                responses, [window_size // 2, window_size // 2], mode="reflect"
            )
            mean = (
                moving_median(pad, window_size)
                if use_median
                else moving_mean(pad, window_size)
            )
            mean = mean[: responses.size]
            std = moving_std(pad, window_size)[: responses.size]

        ld = mean + std * sigma
        return cls(
            mean,
            ld,
            ld,
            name="Gaussian" + " Median" if use_median else "",
            params={"sigma": sigma},
            window_size=window_size,
        )

    @classmethod
    def fromPoisson(
        cls,
        responses: np.ndarray,
        alpha: float = 0.05,
        beta: float = 0.05,
        window_size: int = 0,
        use_median: bool = False,
    ) -> "SPCalLimit":
        if responses.size == 0:
            raise ValueError("fromPoisson: responses is size 0")

        if window_size == 0:  # No window
            mean = np.median(responses) if use_median else np.mean(responses)
        else:
            pad = np.pad(
                responses, [window_size // 2, window_size // 2], mode="reflect"
            )
            mean = (
                moving_median(pad, window_size)
                if use_median
                else moving_mean(pad, window_size)
            )
            mean = mean[: responses.size]

        sc, sd = poisson_limits(mean, alpha=alpha, beta=beta)

        return cls(
            mean,
            mean + sc,
            mean + sd,
            name="Poisson" + " Median" if use_median else "",
            params={"alpha": alpha, "beta": beta},
            window_size=window_size,
        )

    @classmethod
    def fromBest(
        cls,
        responses: np.ndarray,
        sigma: float = 3.0,
        alpha: float = 0.05,
        beta: float = 0.05,
        window_size: int = 0,
        use_median: bool = False,
    ) -> "SPCalLimit":
        mean = np.median(responses) if use_median else np.mean(responses)

        if mean > 50.0:
            return SPCalLimit.fromGaussian(
                responses, sigma=sigma, window_size=window_size, use_median=use_median
            )
        else:
            return SPCalLimit.fromPoisson(
                responses,
                alpha=alpha,
                beta=beta,
                window_size=window_size,
                use_median=use_median,
            )

    @classmethod
    def fromHighest(
        cls,
        responses: np.ndarray,
        sigma: float = 3.0,
        alpha: float = 0.05,
        beta: float = 0.05,
        window_size: int = 0,
        use_median: bool = False,
    ) -> "SPCalLimit":
        mean = np.median(responses) if use_median else np.mean(responses)

        gaussian = SPCalLimit.fromGaussian(
            responses, sigma=sigma, window_size=window_size, use_median=use_median
        )
        poisson = SPCalLimit.fromPoisson(
            responses,
            alpha=alpha,
            beta=beta,
            window_size=window_size,
            use_median=use_median,
        )
        if np.mean(gaussian.mean_background) > np.mean(poisson.mean_background):
            return gaussian
        else:
            return poisson


class SPCalResult(object):
    def __init__(
        self,
        file: str,
        responses: np.ndarray,
        detections: np.ndarray,
        labels: np.ndarray,
        inputs_kws: Dict[str, float],
    ):
        self.file = file

        self.responses = responses
        self.detections = {"signal": detections}
        self.indicies = np.flatnonzero(detections)

        self.background = np.mean(responses[labels == 0])
        self.background_error = np.std(responses[labels == 0])

        self.inputs = inputs_kws

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
