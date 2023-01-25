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
        max_iters: int = 100,
    ) -> "SPCalLimit":
        method = method.lower()
        if method in ["automatic", "best"]:
            return SPCalLimit.fromBest(
                responses, alpha=alpha, window_size=window_size, max_iters=max_iters
            )
        elif method == "highest":
            return SPCalLimit.fromHighest(
                responses, alpha=alpha, window_size=window_size
            )
        elif method.startswith("gaussian"):
            return SPCalLimit.fromGaussian(
                responses, alpha=alpha, window_size=window_size, max_iters=max_iters
            )
        elif method.startswith("poisson"):
            return SPCalLimit.fromPoisson(
                responses, alpha=alpha, window_size=window_size, max_iters=max_iters
            )
        else:
            raise ValueError("fromMethodString: unknown method")

    @classmethod
    def fromGaussian(
        cls,
        responses: np.ndarray,
        alpha: float = 0.001,
        window_size: int = 0,
        max_iters: int = 100,
    ) -> "SPCalLimit":

        if responses.size == 0:
            raise ValueError("fromGaussian: responses is size 0")

        z = NormalDist().inv_cdf(1.0 - alpha / 2.0)  # div 2.0 as one-sided

        threshold, prev_threshold = 0.0, np.inf
        iters = 0
        while np.all(prev_threshold > threshold) and iters < max_iters:
            prev_threshold = threshold
            if window_size == 0:  # No window
                mu = np.median(responses)  # Median is a better estimator of center
                std = np.std(responses)
            else:
                pad = np.pad(
                    responses, [window_size // 2, window_size // 2], mode="reflect"
                )
                mu = moving_median(pad, window_size)
                std = moving_std(pad, window_size)

            threshold = mu + std * z
            iters += 1

        return cls(
            mu,
            threshold,
            name="Gaussian",
            params={"alpha": alpha},
            window_size=window_size,
        )

    @classmethod
    def fromPoisson(
        cls,
        responses: np.ndarray,
        alpha: float = 0.001,
        window_size: int = 0,
        max_iters: int = 100,
    ) -> "SPCalLimit":
        if responses.size == 0:
            raise ValueError("fromPoisson: responses is size 0")

        threshold, prev_threshold = 0.0, np.inf
        iters = 0
        while np.all(prev_threshold > threshold) and iters < max_iters:
            prev_threshold = threshold
            if window_size == 0:  # No window
                mu = np.mean(responses)
            else:
                pad = np.pad(
                    responses, [window_size // 2, window_size // 2], mode="reflect"
                )
                mu = moving_mean(pad, window_size)

            sc, _ = poisson_limits(mu, alpha=alpha)
            threshold = mu + sc
            iters += 1

        return cls(
            mu,
            threshold,
            name="Poisson",
            params={"alpha": alpha},
            window_size=window_size,
        )

    @classmethod
    def fromBest(
        cls,
        responses: np.ndarray,
        alpha: float = 0.001,
        window_size: int = 0,
        max_iters: int = 100,
    ) -> "SPCalLimit":
        mean = np.mean(responses)

        # Todo check for normality
        if mean > 50.0:
            return SPCalLimit.fromGaussian(
                responses, alpha=alpha, window_size=window_size, max_iters=max_iters
            )
        else:
            return SPCalLimit.fromPoisson(
                responses, alpha=alpha, window_size=window_size, max_iters=max_iters
            )

    @classmethod
    def fromHighest(
        cls, responses: np.ndarray, alpha: float = 0.001, window_size: int = 0
    ) -> "SPCalLimit":
        gaussian = SPCalLimit.fromGaussian(
            responses, alpha=alpha, window_size=window_size, max_iters=1
        )
        poisson = SPCalLimit.fromPoisson(
            responses, alpha=alpha, window_size=window_size, max_iters=1
        )
        if np.mean(gaussian.detection_threshold) > np.mean(poisson.detection_threshold):
            return gaussian
        else:
            return poisson
