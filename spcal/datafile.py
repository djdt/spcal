from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable

import numpy as np
import bottleneck as bn

from spcal.detection import accumulate_detections, combine_detections

import logging

from spcal import poisson
from spcal.dists.util import compound_poisson_lognormal_quantile_lookup

logger = logging.getLogger(__name__)


class SPCalDataFile(object):
    def __init__(
        self,
        path: Path,
        signals: np.ndarray,
        isotopes: np.ndarray,
        event_time: float,
        times: np.ndarray | None = None,
        instrument_type: str | None = None,
    ):
        self.path = path
        self.instrument_type = instrument_type

        self.signals = signals
        self.isotopes = isotopes

        self.event_time = event_time

        self._times = times

    @property
    def masses(self) -> np.ndarray:
        return self.isotopes["Mass"]

    @property
    def names(self) -> list[str]:
        return [f"{isotope['Symbol']}{isotope['Isotope']}" for isotope in self.isotopes]

    @property
    def num_events(self) -> int:
        return self.signals.shape[0]

    @property
    def num_isotopes(self) -> int:
        return self.signals.shape[1]

    @property
    def times(self) -> np.ndarray:
        if self._times is None:
            self._times = np.arange(self.num_events) * self.event_time
        return self._times

    def indexForName(self, name: str) -> int:
        return self.names.index(name)

    def indexForIsotope(self, isotope: np.ndarray) -> int:
        index = np.flatnonzero(
            np.logical_and(
                self.isotopes["Symbol"] == isotope["Symbol"],
                self.isotopes["Isotope"] == isotope["Isotope"],
            )
        )
        assert index.size == 1
        return index[0]

    def isTOF(self) -> bool:
        return self.instrument_type == "TOF"


class SPCalLimit(object):
    ITER_EPS = 1e-2

    def __init__(
        self,
        name: str,
        mean_signal: float | np.ndarray | None = None,
        detection_threshold: float | np.ndarray | None = None,
        signals: np.ndarray | None = None,
        window_size: int = 0,
        max_iterations: int = 1,
    ):
        self.name = name

        self.window_size = window_size
        self.max_iterations = max_iterations

        self.iterations_required = 0

        if mean_signal is not None and detection_threshold is not None:
            self.mean_signal = mean_signal
            self.detection_threshold = detection_threshold
        elif signals is not None:
            self.mean_signal, self.detection_threshold = self.calculate(signals)
        else:
            raise ValueError(
                "either mean_signal and detection_threshold or signals must be provided"
            )

    def __str__(self) -> str:
        return (
            f"SPCalLimit({self.name}, mean={self.mean_signal:.4g}, "
            f"threshold={self.detection_threshold:.4g})"
        )

    def thresholdFunction(self, signals: np.ndarray) -> tuple[float, float]:
        raise NotImplementedError

    def windowedThresholdFunction(
        self, signals: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def calculate(
        self, signals: np.ndarray
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        mu, threshold, prev_threshold = 0.0, np.inf, np.inf
        self.iterations_required = 0
        if self.window_size != 0:
            halfwin = self.window_size // 2
            padded_signal = np.pad(signals, [halfwin, halfwin], mode="reflect")

        while (
            np.all(np.abs(prev_threshold - threshold) > SPCalLimit.ITER_EPS)
            and self.iterations_required < self.max_iterations
        ) or self.iterations_required == 0:
            prev_threshold = threshold

            if self.window_size == 0:
                mu, threshold = self.thresholdFunction(signals[signals < threshold])
            else:
                mu, threshold = self.windowedThresholdFunction(
                    np.where(padded_signal > threshold, padded_signal, np.nan),  # type: ignore , is bound
                )

            self.iterations_required += 1

        if (
            self.iterations_required == self.max_iterations and self.max_iterations != 1
        ):  # pragma: no cover
            logger.warning(f"reached iteration of {self.max_iterations}")

        return mu, threshold


class SPCalGaussianLimit(SPCalLimit):
    def __init__(
        self,
        signals: np.ndarray,
        alpha: float = 1e-3,
        window_size: int = 0,
        max_iterations: int = 1,
    ):
        self.z = NormalDist().inv_cdf(1.0 - alpha)
        self.alpha = alpha

        super().__init__(
            "Gaussian",
            signals=signals,
            window_size=window_size,
            max_iterations=max_iterations,
        )

    @staticmethod
    def isGaussianDistributed(signals: np.ndarray) -> bool:
        nonzero_response = signals > 0.0
        nonzero_count = np.count_nonzero(nonzero_response)
        low_signals = signals[np.logical_and(nonzero_response, signals <= 5.0)]
        # Less than 5% of nonzero values are below 5, equivilent to background of ~ 10
        return bool(nonzero_count > 0 and low_signals.size / nonzero_count < 0.05)

    def thresholdFunction(self, signals: np.ndarray) -> tuple[float, float]:
        mu = bn.nanmean(signals)
        return mu, mu + bn.nanstd(signals) * self.z

    def windowedThresholdFunction(
        self, signals: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        halfwin = self.window_size // 2
        mu = bn.move_mean(signals, self.window_size, min_count=halfwin + 1)[
            2 * halfwin :
        ]
        std = bn.move_std(signals, self.window_size, min_count=halfwin + 1)[
            2 * halfwin :
        ]
        return mu, mu + std * self.z


class SPCalPoissonLimit(SPCalLimit):
    FUNCTIONS = {
        "currie": poisson.currie,
        "formula a": poisson.formula_a,
        "formula c": poisson.formula_c,
        "stapleton": poisson.stapleton_approximation,
    }

    def __init__(
        self,
        signals: np.ndarray,
        function: str = "formula c",
        alpha: float = 1e-3,
        window_size: int = 0,
        max_iterations: int = 1,
    ):
        if function not in SPCalPoissonLimit.FUNCTIONS.keys():
            raise ValueError(
                "fomula must be one of", ", ".join(SPCalPoissonLimit.FUNCTIONS.keys())
            )

        self.function = function
        self.alpha = alpha

        self.beta = 0.05
        self.t_sample = 1.0
        self.t_blank = 1.0
        self.eta = 2.0
        self.epsilon = 0.5

        super().__init__(
            "Poisson",
            signals=signals,
            window_size=window_size,
            max_iterations=max_iterations,
        )

    def poissonFunctionArgs(self) -> dict:
        if self.function == "currie":
            return {"beta": self.beta, "eta": self.eta, "epsilon": self.epsilon}
        else:
            return {
                "beta": self.beta,
                "t_sample": self.t_sample,
                "t_blank": self.t_blank,
            }

    def thresholdFunction(self, signals: np.ndarray) -> tuple[float, float]:
        mu = bn.nanmean(signals)
        sc, _ = self.FUNCTIONS[self.function](
            mu, alpha=self.alpha, **self.poissonFunctionArgs()
        )
        return mu, np.ceil(mu + sc)

    def windowedThresholdFunction(
        self, signals: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        halfwin = self.window_size // 2
        mu = bn.move_mean(signals, self.window_size, min_count=halfwin + 1)[
            2 * halfwin :
        ]
        sc, _ = self.FUNCTIONS[self.function](
            mu, alpha=self.alpha, **self.poissonFunctionArgs()
        )
        return mu, np.ceil(mu + sc)


class SPCalCompoundPoissonLimit(SPCalLimit):
    def __init__(
        self,
        signals: np.ndarray,
        sigma: float = 0.5,
        alpha: float = 1e-3,
        window_size: int = 0,
        max_iterations: int = 1,
    ):
        self.alpha = alpha
        self.sigma = sigma
        super().__init__(
            "CompoundPoisson",
            signals=signals,
            window_size=window_size,
            max_iterations=max_iterations,
        )

    def thresholdFunction(self, signals: np.ndarray) -> tuple[float, float]:
        lam = bn.nanmean(signals)
        mu = np.log(lam) - 0.5 * self.sigma**2
        return lam, float(
            compound_poisson_lognormal_quantile_lookup(
                1.0 - self.alpha, lam, mu, self.sigma
            )
        )

    def windowedThresholdFunction(
        self, signals: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        halfwin = self.window_size // 2
        lam = bn.move_mean(signals, self.window_size, min_count=halfwin + 1)[
            2 * halfwin :
        ]
        mu = np.full_like(lam, -0.5 * self.sigma**2)
        return lam, compound_poisson_lognormal_quantile_lookup(  # type: ignore , is array
            1.0 - self.alpha, lam, mu, np.full_like(lam, self.sigma)
        )


class SPCalProcessingResult(object):
    def __init__(
        self,
        signal: np.ndarray,
        limit: SPCalLimit,
        detections: np.ndarray,
        labels: np.ndarray,
        regions: np.ndarray,
        indicies: np.ndarray | None = None,
    ):
        self.signal = signal
        self.detections = detections
        self.labels = labels
        self.regions = regions

        self.indicies = indicies

    @property
    def num_events(self) -> int:
        return self.signal.shape[0]


class SPCalProcessingMethod(object):
    def __init__(
        self,
        limit_method: str = "automatic",
        limit_accumulation_method: str = "signal mean",
        gaussian_kws: dict[str, Any] | None = None,
        poisson_kws: dict[str, Any] | None = None,
        compound_poisson_kws: dict[str, Any] | None = None,
        window_size: int = 0,
        max_iterations: int = 1,
        points_required: int = 1,
        prominence_required: float = 0.2,
    ):
        # deafult kws
        if gaussian_kws is None:
            gaussian_kws = {"alpha": 2.867e-7, "windows_size": 0}
        if poisson_kws is None:
            poisson_kws = {"alpha": 1e-3, "function": "formula c"}
        if compound_poisson_kws is None:
            compound_poisson_kws = {"alpha": 1e-6, "sigma": 0.5}

        self.limit_method = limit_method
        self.window_size = window_size
        self.max_iterations = max_iterations

        self.gaussian_kws = gaussian_kws
        self.poisson_kws = poisson_kws
        self.compound_poisson_kws = compound_poisson_kws

        self.accumulation_method = limit_accumulation_method
        self.points_required = points_required
        self.prominence_required = prominence_required

        self.sigmas = None

    # def assignCombinedIndicies(self, results: list[SPCalProcessingResult]) -> None:
    #     regions = [result.regions for result in results]
    #     all_regions = ext.combine_regions(regions, 2)
    #
    #     for result in results:
    #         idx = np.searchsorted(all_regions[:, 0], result.regions[:, 1], side="left") - 1
    #         result.

    def limitsForIsotope(
        self, data_file: SPCalDataFile, isotope: np.ndarray
    ) -> SPCalLimit:
        signals = data_file.signals[:, data_file.indexForIsotope(isotope)]
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
                logger.warning("Compound-Poisson limit created for non TOF data file")
            return SPCalCompoundPoissonLimit(
                signals,
                **self.compound_poisson_kws,
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

        for isotope in data_file.isotopes:
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
                data_file.signals[:, data_file.indexForIsotope(isotope)],
                limit_accumulation=limit_accumulation,
                limit_detection=limit_detection,
                points_required=self.points_required,
                prominence_required=self.prominence_required,
            )

            result = SPCalProcessingResult(
                data_file.signals[:, data_file.indexForIsotope(isotope)],
                limit,
                detections,
                labels,
                regions,
            )
            results[isotope] = result

        return results
