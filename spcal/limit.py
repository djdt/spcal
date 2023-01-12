from statistics import NormalDist
from typing import Dict

import numpy as np

from spcal.calc import moving_mean, moving_median, moving_std
from spcal.poisson import formula_c as poisson_limits


class SPCalLimit(object):
    def __init__(
        self,
        mean_background: float | np.ndarray,
        detection_threshold: float | np.ndarray,
        name: str,
        params: Dict[str, float],
        window_size: int = 0,
    ):
        self.mean_background = mean_background
        self.detection_threshold = detection_threshold

        self.name = name
        self.params = params
        self.window_size = window_size

    @classmethod
    def fromMethodString(
        cls,
        method: str,
        responses: np.ndarray,
        alpha: float = 0.001,
        window_size: int = 0,
    ) -> "SPCalLimit":
        method = method.lower()
        if method in ["automatic", "best"]:
            return SPCalLimit.fromBest(responses, alpha=alpha, window_size=window_size)
        elif method == "highest":
            return SPCalLimit.fromHighest(
                responses, alpha=alpha, window_size=window_size
            )
        elif method.startswith("gaussian"):
            return SPCalLimit.fromGaussian(
                responses,
                alpha=alpha,
                window_size=window_size,
                use_median="median" in method,
            )
        elif method.startswith("poisson"):
            return SPCalLimit.fromPoisson(
                responses,
                alpha=alpha,
                window_size=window_size,
                use_median="median" in method,
            )
        else:
            raise ValueError("fromMethodString: unknown method")

    @classmethod
    def fromGaussian(
        cls,
        responses: np.ndarray,
        alpha: float = 0.001,
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

        z = NormalDist().inv_cdf(1.0 - alpha)

        return cls(
            mean,
            mean + std * z,
            name="Gaussian" + (" Median" if use_median else ""),
            params={"alpha": alpha},
            window_size=window_size,
        )

    @classmethod
    def fromPoisson(
        cls,
        responses: np.ndarray,
        alpha: float = 0.001,
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

        sc, _ = poisson_limits(mean, alpha=alpha)

        return cls(
            mean,
            mean + sc,
            name="Poisson" + (" Median" if use_median else ""),
            params={"alpha": alpha},
            window_size=window_size,
        )

    @classmethod
    def fromBest(
        cls,
        responses: np.ndarray,
        alpha: float = 0.001,
        window_size: int = 0,
        use_median: bool = False,
    ) -> "SPCalLimit":
        mean = np.median(responses) if use_median else np.mean(responses)

        # Todo check for normality
        if mean > 50.0:
            return SPCalLimit.fromGaussian(
                responses, alpha=alpha, window_size=window_size, use_median=use_median
            )
        else:
            return SPCalLimit.fromPoisson(
                responses, alpha=alpha, window_size=window_size, use_median=use_median
            )

    @classmethod
    def fromHighest(
        cls,
        responses: np.ndarray,
        alpha: float = 0.001,
        window_size: int = 0,
        use_median: bool = False,
    ) -> "SPCalLimit":
        gaussian = SPCalLimit.fromGaussian(
            responses, alpha=alpha, window_size=window_size, use_median=use_median
        )
        poisson = SPCalLimit.fromPoisson(
            responses, alpha=alpha, window_size=window_size, use_median=use_median
        )
        if np.mean(gaussian.detection_threshold) > np.mean(poisson.detection_threshold):
            return gaussian
        else:
            return poisson
