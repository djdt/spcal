"""Helper class for SPCal limits and thresholding."""
import logging
from statistics import NormalDist
from typing import Dict

import bottleneck as bn
import numpy as np

from spcal.poisson import formula_c

logger = logging.getLogger(__name__)


class SPCalLimit(object):
    poisson_formula = formula_c

    """Limit and threshold class.

    Limits should be created through the class methods ``fromMethodString``,
    ``fromBest``, ``fromHighest``, ``fromGaussian`` and ``fromPoisson``.

    These functions will iteratively threshold for ``max_iters`` until a stable
    threshold is reached. Iteration can help threshold in samples with a very large
    number of particles to background. For no iteration, pass 0 to ``max_iters``.

    Windowed thresholding can be performed by passing a number greater than 0 to
    ``window_size``. This will calculate the signal mean and threshold for every
    point using data from only the surrounding window.

    Attributes:
        mean_signal: average signal
        detection_threshold: threshold for particle detection
        name: name of the filter / method
        params: filter / method parameters
        window_size: size of window used or 0 if unwindowed
    """

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
        """Takes a string and returns limit class.

        Valid stings are 'automatic', 'best', 'highest', 'gaussian', 'poisson'.

        Args:
            method: method to use
            responses: single particle data
            poisson_alpha: error rate for Poisson thresholding
            gaussian_alpha: error rate for Gaussian thresholding
            window_size: size of window to use, 0 for no window
            max_iters: maximum iterations to try
        """
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
    def fromCompoundPoisson(
        cls,
        responses: np.ndarray,
        sis: float | np.ndarray,
        n_accumulations: int,
        alpha: float = 0.001,
        size: int = 10000,
    ) -> "SPCalLimit":
        lam = responses.mean()

        if isinstance(sis, float):  # passed average, give an estiamtion
            sis = np.random.normal(sis, sis, size=100)

        sim = np.zeros(size)
        for _ in range(n_accumulations):
            poi = np.random.poisson(lam / n_accumulations, size=size)
            unique, idx, counts = np.unique(
                poi, return_counts=True, return_inverse=True
            )
            for i, (u, c) in enumerate(zip(unique, counts)):
                sim[idx == i] += np.sum(np.random.choice(sis, size=(u, c)), axis=0)

        sim /= np.mean(sis)
        return SPCalLimit(
            lam,
            float(np.quantile(sim, alpha)),
            name="CompoundPoisson",
            params={"alpha": alpha},
            window_size=0,
        )

    @classmethod
    def fromGaussian(
        cls,
        responses: np.ndarray,
        alpha: float = 1e-6,
        window_size: int = 0,
        max_iters: int = 10,
    ) -> "SPCalLimit":
        """Gaussian thresholding.

        Threshold is calculated as the mean + z * std deviation of the sample.

        Args:
            responses: single-particle data
            alpha: type I error rate
            window_size: size of window, 0 for no window
            max_iters: max iterations, 0 for no iteration
        """
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
        """Poisson thresholding.

        Uses Formula C from the MARLAP manual to calculate the limit of criticality.

        Args:
            responses: single-particle data
            alpha: type I error rate
            window_size: size of window, 0 for no window
            max_iters: max iterations, 0 for no iteration
        """
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

            sc, _ = SPCalLimit.poisson_formula(mu, alpha=alpha)
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
        """Returns 'best' threshold.

        Uses a Poisson threshold to calculate the mean of the background (signal below
        the limit of criticality). If this is above 10.0 then Gaussian thresholding is
        used instead.

        Args:
            responses: single-particle data
            poisson_alpha: type I error rate for Poisson
            gaussian_alpha: type I error rate for Gaussian
            window_size: size of window, 0 for no window
            max_iters: max iterations, 0 for no iteration
        """
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
        """Returns highest threshold.

        Calculates the Poisson and Gaussian thresholds and returns on with the highest
        detection threshold.

        Args:
            responses: single-particle data
            poisson_alpha: type I error rate for Poisson
            gaussian_alpha: type I error rate for Gaussian
            window_size: size of window, 0 for no window
            max_iters: max iterations, 0 for no iteration
        """
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
