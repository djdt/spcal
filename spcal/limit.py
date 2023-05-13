"""Helper class for SPCal limits and thresholding."""
import logging
from statistics import NormalDist
from typing import Callable, Dict, Tuple

import bottleneck as bn
import numpy as np

from spcal.poisson import currie, formula_a, formula_c, stapleton_approximation

logger = logging.getLogger(__name__)


class SPCalLimit(object):
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
    """

    def __init__(
        self,
        mean_background: float | np.ndarray,
        detection_threshold: float | np.ndarray,
        name: str,
        params: Dict[str, float],
    ):
        self.mean_signal = mean_background
        self.detection_threshold = detection_threshold

        self.name = name
        self.params = params

    def __str__(self) -> str:
        pstring = ";".join(f"{k}={v}" for k, v in self.params.items() if v != 0)
        return f"{self.name} ({pstring})" if len(pstring) > 0 else self.name

    @classmethod
    def fromMethodString(
        cls,
        method: str,
        responses: np.ndarray,
        poisson_kws: dict | None = None,
        gaussian_kws: dict | None = None,
        compound_kws: dict | None = None,
        window_size: int = 0,
        max_iters: int = 1,
    ) -> "SPCalLimit":
        """Takes a string and returns limit class.

        Valid stings are 'automatic', 'best', 'highest', 'compound', gaussian' and
        'poisson'.

        The CompoundPoisson method is seeded with a set number so will always give
        the same results.

        Args:
            method: method to use
            responses: single particle data
            compound_kws: key words for Compound Poisson thresholding
            poisson_kws: key words for Poisson thresholding
            gaussian_kws: key words for Gaussian thresholding
            window_size: size of window to use, 0 for no window
            max_iters: maximum iterations to try
        """
        if compound_kws is None:
            compound_kws = {}
        if gaussian_kws is None:
            gaussian_kws = {}
        if poisson_kws is None:
            poisson_kws = {}

        method = method.lower()
        if method in ["automatic", "best"]:
            return SPCalLimit.fromBest(
                responses,
                poisson_kws=poisson_kws,
                gaussian_kws=gaussian_kws,
                window_size=window_size,
                max_iters=max_iters,
            )
        elif method == "highest":
            return SPCalLimit.fromHighest(
                responses,
                poisson_kws=poisson_kws,
                gaussian_kws=gaussian_kws,
                window_size=window_size,
                max_iters=max_iters,
            )
        elif method.startswith("compound"):
            return SPCalLimit.fromCompoundPoisson(
                responses,
                alpha=compound_kws.get("alpha", 1e-6),
                single_ion=compound_kws.get("single ion", 1.0),
                accumulations=compound_kws.get("accumulations", 1),
                max_iters=max_iters,
                seed=294879019,  # use a seed for consitent results
            )
        elif method.startswith("gaussian"):
            return SPCalLimit.fromGaussian(
                responses,
                alpha=gaussian_kws.get("alpha", 1e-6),
                window_size=window_size,
                max_iters=max_iters,
            )
        elif method.startswith("poisson"):
            return SPCalLimit.fromPoisson(
                responses,
                alpha=poisson_kws.get("alpha", 0.001),
                formula=poisson_kws.get("formula", "formula c"),
                formula_kws=poisson_kws.get("params", None),
                window_size=window_size,
                max_iters=max_iters,
            )
        else:
            raise ValueError("fromMethodString: unknown method")

    @classmethod
    def fromCompoundPoisson(
        cls,
        responses: np.ndarray,
        single_ion: float | np.ndarray,
        accumulations: int,
        alpha: float = 0.001,
        max_iters: int = 1,
        size: int | None = None,
        seed: int | None = None,
    ) -> "SPCalLimit":
        """Calculate threshold from simulated compound distribution.

        ToF data is a the sum of multiple Poisson accumulation events, each of which are
        an independant sample of lognormal like SIS distribution. This function will
        simulate the expected background and calculate the appropriate quantile for a
        given alpha value.

        Args:
            responses: single-particle data
            single_ion: single ion area as an average, distribution or histogram of
                stacked bins and counts
            size: size of simulation
            accumulations: number of accumulation per acquisition
            alpha: type I error rate
            max_iters: number of iterations, set to 1 for no iters
            size: size of simulation, larger values will give more consistent quantiles
            seed: seed for random number generator

        References:
            Gundlach-Graham, A.; Lancaster, R. Mass-Dependent Critical Value Expressions
                for Particle Finding in Single-Particle ICP-TOFMS, Anal. Chem 2023
                https://doi.org/10.1021/acs.analchem.2c05243
            Gershman, D.; Gliese, U.; Dorelli, J.; Avanov, L.; Barrie, A.; Chornay, D.;
                MacDonald, E.; Hooland, M.l Giles, B.; Pollock, C. The parameterization
                of microchannel-plate-based detection systems, J. Geo. R. 2018
                https://doi.org/10.1002/2016JA022563
        """

        rng = np.random.default_rng(seed=seed)
        if size is None:
            size = responses.size

        # If given a float then generate a distribution with estimated params
        weights = None
        if isinstance(single_ion, float):
            average_single_ion = single_ion
            single_ion = rng.lognormal(np.log(single_ion), 0.5, size=10000)
        else:
            if single_ion.ndim == 2:  # histogram of (bins, counts)
                weights = single_ion[:, 1] / single_ion[:, 1].sum()
                single_ion = single_ion[:, 0]
            average_single_ion = np.average(single_ion, weights=weights)

        threshold, prev_threshold = np.inf, np.inf
        iters = 0
        while (np.all(prev_threshold > threshold) and iters < max_iters) or iters == 0:
            prev_threshold = threshold

            lam = bn.nanmean(responses[responses < threshold])

            comp = rng.poisson(lam, size=size) * rng.choice(
                single_ion, size=size, p=weights
            )

            comp /= average_single_ion
            threshold = float(np.quantile(comp, 1.0 - alpha))
            iters += 1

        if iters == max_iters and max_iters != 1:  # pragma: no cover
            logger.warning("fromCompoundPoisson: reached max_iters")

        return cls(
            lam,
            threshold,
            name="CompoundPoisson",
            params={"alpha": alpha, "iters": iters - 1},
        )

    @classmethod
    def fromGaussian(
        cls,
        responses: np.ndarray,
        alpha: float = 1e-6,
        window_size: int = 0,
        max_iters: int = 1,
    ) -> "SPCalLimit":
        """Gaussian thresholding.

        Threshold is calculated as the mean + z * std deviation of the sample.

        Args:
            responses: single-particle data
            alpha: type I error rate
            window_size: size of window, 0 for no window
            max_iters: max iterations, 1 for no iteration
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

            threshold = mu + std * z
            iters += 1

        if iters == max_iters and max_iters != 1:  # pragma: no cover
            logger.warning("fromPoisson: reached max_iters")

        return cls(
            mu,
            threshold,
            name="Gaussian",
            params={"alpha": alpha, "window": window_size, "iters": iters - 1},
        )

    @classmethod
    def fromPoisson(
        cls,
        responses: np.ndarray,
        alpha: float = 0.001,
        formula: str = "formula c",
        formula_kws: Dict[str, float] | None = None,
        window_size: int = 0,
        max_iters: int = 1,
    ) -> "SPCalLimit":
        """Poisson thresholding.

        Calculate the limit of criticality using the supplied formula and params.

        Args:
            responses: single-particle data
            alpha: type I error rate
            formula: formula to use, {currie, formula a, formula c, stapleton}
            formula_kws: kws for formula
            window_size: size of window, 0 for no window
            max_iters: max iterations, 1 for no iteration
        """
        if responses.size == 0:  # pragma: no cover
            raise ValueError("fromPoisson: responses is size 0")

        formula = formula.lower()
        if formula == "currie":
            poisson_fn: Callable[
                [...], Tuple[float | np.ndarray, float | np.ndarray]
            ] = currie
        elif formula == "formula a":
            poisson_fn = formula_a
        elif formula == "formula c":
            poisson_fn = formula_c
        elif formula.startswith("stapleton"):
            poisson_fn = stapleton_approximation
        else:
            raise ValueError(f"unknown poisson limit formula: {formula}")

        if formula_kws is None:
            formula_kws = {}
        formula_kws["alpha"] = alpha

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

            sc, _ = poisson_fn(mu, **formula_kws)
            threshold = np.ceil(mu + sc)
            iters += 1

        if iters == max_iters and max_iters != 1:  # pragma: no cover
            logger.warning("fromPoisson: reached max_iters")

        return cls(
            mu,
            threshold,
            name="Poisson",
            params={"alpha": alpha, "window": window_size, "iters": iters - 1},
        )

    @classmethod
    def fromBest(
        cls,
        responses: np.ndarray,
        poisson_kws: dict | None = None,
        gaussian_kws: dict | None = None,
        window_size: int = 0,
        max_iters: int = 1,
    ) -> "SPCalLimit":
        """Returns 'best' threshold.

        Uses a Poisson threshold to calculate the mean of the background (signal below
        the limit of criticality). If this is above 10.0 then Gaussian thresholding is
        used instead.

        Args:
            responses: single-particle data
            poisson_kws: keywords for Poisson
            gaussian_kws: keywords for Gaussian
            window_size: size of window, 0 for no window
            max_iters: max iterations, 0 for no iteration
        """
        if poisson_kws is None:
            poisson_kws = {}
        if gaussian_kws is None:
            gaussian_kws = {}
        # Check that the non-detection region is normalish (λ > 10)
        poisson = SPCalLimit.fromPoisson(
            responses,
            alpha=poisson_kws.get("alpha", 0.001),
            formula=poisson_kws.get("formula", "formula c"),
            formula_kws=poisson_kws.get("params", None),
            window_size=window_size,
            max_iters=max_iters,
        )
        if np.mean(responses[responses < poisson.detection_threshold]) < 10.0:
            return poisson
        else:
            return SPCalLimit.fromGaussian(
                responses,
                alpha=gaussian_kws.get("alpha", 1e-6),
                window_size=window_size,
                max_iters=max_iters,
            )

    @classmethod
    def fromHighest(
        cls,
        responses: np.ndarray,
        poisson_kws: dict | None = None,
        gaussian_kws: dict | None = None,
        window_size: int = 0,
        max_iters: int = 1,
    ) -> "SPCalLimit":
        """Returns highest threshold.

        Calculates the Poisson and Gaussian thresholds and returns on with the highest
        detection threshold.

        Args:
            responses: single-particle data
            poisson_kws: keywords for Poisson
            gaussian_kws: keywords for Gaussian
            window_size: size of window, 0 for no window
            max_iters: max iterations, 0 for no iteration
        """
        if poisson_kws is None:
            poisson_kws = {}
        if gaussian_kws is None:
            gaussian_kws = {}
        poisson = SPCalLimit.fromPoisson(
            responses,
            alpha=poisson_kws.get("alpha", 0.001),
            formula=poisson_kws.get("formula", "formula c"),
            formula_kws=poisson_kws.get("params", None),
            window_size=window_size,
            max_iters=max_iters,
        )
        gaussian = SPCalLimit.fromGaussian(
            responses,
            alpha=gaussian_kws.get("alpha", 1e-6),
            window_size=window_size,
            max_iters=max_iters,
        )
        if np.mean(gaussian.detection_threshold) > np.mean(poisson.detection_threshold):
            return gaussian
        else:
            return poisson
