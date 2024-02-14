"""Functions for screening data for interesting signals."""


import numpy as np

from spcal.detection import accumulate_detections
from spcal.limit import SPCalLimit


def screen_element(
    x: np.ndarray,
    limit: SPCalLimit | None = None,
    limit_kws: dict | None = None,
    mode: str = "events",
) -> int:
    """Screen element for signals.

    Returns number of points or particles greater than provided limits.
    If no limit is provided then SPCalLimit.fromBest is used with the supplied
    ``limit_kws``.

    Args:
        x: data
        limits: pre-calculated limit
        limit_kws: kwargs for SPCalLimit.fromBest if limit is None
        mode: method of detection, 'events' or 'detections'

    Returns:
        number of detections
    """
    if limit is None:
        if limit_kws is None:  # pragma: no cover
            limit_kws = {}
        limit = SPCalLimit.fromBest(x, **limit_kws)

    if mode == "events":
        count = np.count_nonzero(x > limit.detection_threshold)
    elif mode == "detections":
        count = accumulate_detections(
            x,
            np.minimum(limit.mean_signal, limit.detection_threshold),
            limit.detection_threshold,
        )[0].size
    else:  # pragma: no cover
        raise ValueError("screening mode must be 'events' or 'detections'")

    return count


def non_target_screen(
    x: np.ndarray,
    minimum_count_ppm: float,
    limits: list[SPCalLimit] | None = None,
    limit_kws: dict | None = None,
    mode: str = "events",
) -> np.ndarray:
    """Screen data for potential NP signals.

    Finds signals with ``minimum_count_ppm`` ppm points or particles greater than
    provided limits. If no limit is provided then SPCalLimit.fromBest is used with
    the supplied `limit_kws`.

    Args:
        x: data of shape (events, elements)
        minimum_count_ppm: minimum number of points above limit
        limits: pre-calculated limits, shape (elements,)
        limit_kws: kwargs for SPCalLimit.fromBest if limits is None
        mode: method of detection, 'events' or 'detections'

    Returns:
        indices of elements with potential signals
    """

    if limits is None:
        if limit_kws is None:
            limit_kws = {}
        limits = [SPCalLimit.fromBest(x[:, i], **limit_kws) for i in range(x.shape[1])]

    counts = np.array(
        [screen_element(x[:, i], limit=limits[i], mode=mode) for i in range(x.shape[1])]
    )
    ppm = counts * 1e6 / x.shape[0]
    return np.flatnonzero(ppm > minimum_count_ppm)
