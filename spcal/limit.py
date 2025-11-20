"""Helper class for SPCal limits and thresholding."""

import logging
from statistics import NormalDist

import bottleneck as bn
import numpy as np

from spcal import poisson
from spcal.dists.util import compound_poisson_lognormal_quantile_lookup

logger = logging.getLogger(__name__)


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

    def __repr__(self) -> str:
        return (
            f"SPCalLimit({self.name}, mean={np.nanmean(self.mean_signal):.4g}, "
            f"threshold={np.nanmean(self.detection_threshold):.4g})"
        )

    @property
    def parameters(self) -> dict:
        return {}

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
                    np.where(padded_signal < threshold, padded_signal, np.nan),  # type: ignore , is bound
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

    @property
    def parameters(self) -> dict:
        return {"alpha": self.alpha}

    @staticmethod
    def isGaussianDistributed(signals: np.ndarray) -> bool:
        nonzero_signals = signals > 0.0
        nonzero_count = np.count_nonzero(nonzero_signals)
        low_signals = signals[np.logical_and(nonzero_signals, signals <= 5.0)]
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
        beta: float = 0.05,
        t_sample: float = 1.0,
        t_blank: float = 1.0,
        eta: float = 2.0,
        epsilon: float = 0.5,
        window_size: int = 0,
        max_iterations: int = 1,
    ):
        if function not in SPCalPoissonLimit.FUNCTIONS.keys():
            raise ValueError(
                "fomula must be one of", ", ".join(SPCalPoissonLimit.FUNCTIONS.keys())
            )

        self.function = function
        self.alpha = alpha

        self.beta = beta
        self.t_sample = t_sample
        self.t_blank = t_blank
        self.eta = eta
        self.epsilon = epsilon

        super().__init__(
            "Poisson",
            signals=signals,
            window_size=window_size,
            max_iterations=max_iterations,
        )

    @property
    def parameters(self) -> dict:
        params = {"function": self.function, "alpha": self.alpha, "beta": self.beta}
        if self.function == "currie":
            params.update({"eta": self.eta, "epsilon": self.epsilon})
        else:
            params.update({"t_sample": self.t_sample, "t_blank": self.t_blank})
        return params

    def thresholdFunction(self, signals: np.ndarray) -> tuple[float, float]:
        mu = bn.nanmean(signals)
        params = self.parameters
        fn = params.pop("function")
        sc, _ = self.FUNCTIONS[fn](mu, **params)
        return mu, np.ceil(mu + sc)

    def windowedThresholdFunction(
        self, signals: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        halfwin = self.window_size // 2
        mu = bn.move_mean(signals, self.window_size, min_count=halfwin + 1)[
            2 * halfwin :
        ]
        params = self.parameters
        fn = params.pop("function")
        sc, _ = self.FUNCTIONS[fn](mu, **params)
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

    @property
    def parameters(self) -> dict:
        return {"alpha": self.alpha, "sigma": self.sigma}

    def thresholdFunction(self, signals: np.ndarray) -> tuple[float, float]:
        lam = bn.nanmean(signals)
        mu = -0.5 * self.sigma**2
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
        mu = np.full_like(lam, np.log(lam) - 0.5 * self.sigma**2)
        return lam, compound_poisson_lognormal_quantile_lookup(  # type: ignore , is array
            1.0 - self.alpha, lam, mu, np.full_like(lam, self.sigma)
        )
