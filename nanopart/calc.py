import numpy as np

from typing import Tuple

import nanopart


def calculate_limits(
    responses: np.ndarray, method: str, sigma: float = None, epsilon: float = None
) -> Tuple[str, float, float, float]:

    if responses is None or responses.size == 0:
        return

    mean = np.nanmean(responses)
    gaussian = None
    poisson: Tuple[float, float] = None

    if method == "Automatic":
        method = "Poisson" if mean < 50.0 else "Gaussian"

    if method in ["Highest", "Gaussian"]:
        if sigma is not None:
            gaussian = mean + sigma * np.nanstd(responses)

    if method in ["Highest", "Poisson"]:
        if epsilon is not None:
            yc, yd = nanopart.poisson_limits(mean, epsilon=epsilon)
            poisson = (mean + yc, mean + yd)

    if method == "Highest":
        if gaussian is not None and poisson is not None:
            method = "Gaussian" if gaussian > poisson[1] else "Poisson"

    if method == "Gaussian" and gaussian is not None:
        return (method, mean, gaussian, gaussian)
    elif method == "Poisson" and poisson is not None:
        return (method, mean, poisson[0], poisson[1])
    else:
        return None
