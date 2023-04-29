"""Helper class for SPCal limits and thresholding."""
import logging
from statistics import NormalDist
from typing import Dict

import bottleneck as bn
import numpy as np

from spcal.poisson import formula_c as poisson_limits

logger = logging.getLogger(__name__)


class SPCalLimit(object):
    """Limit and threshold class.

    Attributes:
        mean_background: """
    def __init__(
        self,
        mean_background: float | np.ndarray,
        detection_threshold: float | np.ndarray,
        name: str,
        params: Dict[str, float],
        window_size: int = 0,
    ):
        self.mean_signal = mean_background
        self.detection_threshold = detection_threshold

        self.name = name
        self.params = params
        self.window_size = window_size

    def __str__(self) -> str:
        return f"{self.name}: {self.detection_threshold}"

    @classmethod
    def fromMethodString(
        cls,
        method: str,
        responses: np.ndarray,
        poisson_alpha: float = 0.001,
        gaussian_alpha: float = 1e-6,
        window_size: int = 0,
        max_iters: int = 10,
    ) -> "SPCalLimit":
        method = method.lower()
        if method in ["automatic", "best"]:
            return SPCalLimit.fromBest(
                responses,
                poisson_alpha=poisson_alpha,
                gaussian_alpha=gaussian_alpha,
                window_size=window_size,
                max_iters=max_iters,
            )
        elif method == "highest":
            return SPCalLimit.fromHighest(
                responses,
                poisson_alpha=poisson_alpha,
                gaussian_alpha=gaussian_alpha,
                window_size=window_size,
            )
        elif method.startswith("gaussian"):
            return SPCalLimit.fromGaussian(
                responses,
                alpha=gaussian_alpha,
                window_size=window_size,
                max_iters=max_iters,
            )
        elif method.startswith("poisson"):
            return SPCalLimit.fromPoisson(
                responses,
                alpha=poisson_alpha,
                window_size=window_size,
                max_iters=max_iters,
            )
        else:
            raise ValueError("fromMethodString: unknown method")

    @classmethod
    def fromGaussian(
        cls,
        responses: np.ndarray,
        alpha: float = 1e-6,
        window_size: int = 0,
        max_iters: int = 10,
    ) -> "SPCalLimit":
        if responses.size == 0:  # pragma: no cover
            raise ValueError("fromGaussian: responses is size 0")

        z = NormalDist().inv_cdf(1.0 - alpha)

        threshold, prev_threshold = np.inf, np.inf
        iters = 0
        while (np.all(prev_threshold > threshold) and iters < max_iters) or iters == 0:
            prev_threshold = threshold

            if window_size == 0:  # No window
                mu = bn.nanmean(responses[responses < threshold])
                std = bn.nanstd(responses[responses < threshold])
            else:
                halfwin = window_size // 2
                pad = np.pad(
                    np.where(responses < threshold, responses, np.nan),
                    [halfwin, halfwin],
                    mode="reflect",
                )
                mu = bn.move_mean(pad, window_size, min_count=1)[2 * halfwin :]
                std = bn.move_std(pad, window_size, min_count=1)[2 * halfwin :]

            # Consistency with Poisson
            threshold = np.ceil(mu + std * z)
            iters += 1

        if iters == max_iters and max_iters != 1:  # pragma: no cover
            logger.warning("fromPoisson: reached max_iters")

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
        max_iters: int = 10,
    ) -> "SPCalLimit":
        if responses.size == 0:  # pragma: no cover
            raise ValueError("fromPoisson: responses is size 0")

        threshold, prev_threshold = np.inf, np.inf
        iters = 0
        while (np.all(prev_threshold > threshold) and iters < max_iters) or iters == 0:
            prev_threshold = threshold
            if window_size == 0:  # No window
                mu = bn.nanmean(responses[responses < threshold])
            else:
                halfwin = window_size // 2
                pad = np.pad(
                    np.where(responses < threshold, responses, np.nan),
                    [halfwin, halfwin],
                    mode="reflect",
                )
                mu = bn.move_mean(pad, window_size, min_count=1)[2 * halfwin :]

            sc, _ = poisson_limits(mu, alpha=alpha)
            threshold = np.ceil(mu + sc)
            iters += 1

        if iters == max_iters and max_iters != 1:  # pragma: no cover
            logger.warning("fromPoisson: reached max_iters")

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
        poisson_alpha: float = 0.001,
        gaussian_alpha: float = 1e-6,
        window_size: int = 0,
        max_iters: int = 10,
    ) -> "SPCalLimit":
        # Check that the non-detection region is normalish (λ > 10)
        poisson = SPCalLimit.fromPoisson(
            responses,
            alpha=poisson_alpha,
            window_size=window_size,
            max_iters=max_iters,
        )
        if np.mean(responses[responses < poisson.detection_threshold]) < 10.0:
            return poisson
        else:
            return SPCalLimit.fromGaussian(
                responses,
                alpha=gaussian_alpha,
                window_size=window_size,
                max_iters=max_iters,
            )

    @classmethod
    def fromHighest(
        cls,
        responses: np.ndarray,
        poisson_alpha: float = 0.001,
        gaussian_alpha: float = 1e-6,
        window_size: int = 0,
        max_iters: int = 10,
    ) -> "SPCalLimit":
        gaussian = SPCalLimit.fromGaussian(
            responses,
            alpha=gaussian_alpha,
            window_size=window_size,
            max_iters=max_iters,
        )
        poisson = SPCalLimit.fromPoisson(
            responses, alpha=poisson_alpha, window_size=window_size, max_iters=max_iters
        )
        if np.mean(gaussian.detection_threshold) > np.mean(poisson.detection_threshold):
            return gaussian
        else:
            return poisson
