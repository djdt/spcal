from typing import Dict

import numpy as np

from spcal.calc import moving_mean, moving_median, moving_std
from spcal.poisson import formula_c as poisson_limits


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
    def fromMethodString(
        cls,
        method: str,
        responses: np.ndarray,
        sigma: float = 3.0,
        alpha: float = 0.05,
        beta: float = 0.05,
        window_size: int = 0,
    ) -> "SPCalLimit":
        method = method.lower()
        if method in ["automatic", "best"]:
            return SPCalLimit.fromBest(
                responses,
                sigma=sigma,
                alpha=alpha,
                beta=beta,
                window_size=window_size,
            )
        elif method == "highest":
            return SPCalLimit.fromHighest(
                responses,
                sigma=sigma,
                alpha=alpha,
                beta=beta,
                window_size=window_size,
            )
        elif method.startswith("gaussian"):
            return SPCalLimit.fromGaussian(
                responses,
                sigma=sigma,
                window_size=window_size,
                use_median="median" in method,
            )
        elif method.startswith("poisson"):
            return SPCalLimit.fromPoisson(
                responses,
                alpha=alpha,
                beta=beta,
                window_size=window_size,
                use_median="median" in method,
            )
        else:
            raise ValueError("fromMethodString: unknown method")

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
            name="Gaussian" + (" Median" if use_median else ""),
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
            name="Poisson" + (" Median" if use_median else ""),
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

        # Todo check for normality
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
        if np.mean(gaussian.limit_of_detection) > np.mean(poisson.limit_of_detection):
            return gaussian
        else:
            return poisson
