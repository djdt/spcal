"""Functions for detecting and classifying particles."""

import numpy as np

from spcal.lib.spcalext import detection as ext


def accumulate_detections(
    y: np.ndarray,
    limit_accumulation: float | np.ndarray,
    limit_detection: float | np.ndarray,
    points_required: int = 1,
    prominence_required: float = 0.2,
    integrate: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns an array of accumulated detections.

    Peak prominence is calculated for all points above the ``limit_detection``,
    with widths bound by the ``limit_accumulation``.
    Detections are peaks with at least ``points_required`` points above the ``limit_detection``.
    Peaks with overlapping prominences with at least ``prominence_required`` of the maxium height
    will be split.

    Args:
        y: array
        limit_accumulation: minimum accumulation value(s)
        limit_detection: minimum detection value(s)
        points_required: no. points > limit_detection to be detected
        prominence_required: minimum fraction of max prominence for overlapping peaks
        integrate: integrate, otherwise sum

    Returns:
        summed detection regions
        regions [starts, ends]
    """

    if np.any(limit_accumulation > limit_detection):
        raise ValueError("accumulate_detections: limit_accumulation > limit_detection.")
    if points_required < 1:
        raise ValueError("accumulate_detections: minimum size must be >= 1")
    if prominence_required < 0.0 or prominence_required > 1.0:
        raise ValueError(
            "accumulate_detections: prominence_required must be in the range (0.0, 1.0)"
        )

    above = np.greater(y, limit_detection)

    possible_detections = np.flatnonzero(np.logical_and(above, ext.local_maxima(y)))

    prominence, lefts, rights = ext.peak_prominence(
        y, possible_detections, min_base=limit_accumulation
    )

    min_prominence = limit_detection - limit_accumulation

    # First we remove any peaks lower than the lod - base
    if isinstance(min_prominence, np.ndarray):
        detected = prominence >= min_prominence[possible_detections]
    else:
        detected = prominence >= min_prominence
    prominence, lefts, rights = (
        prominence[detected],
        lefts[detected],
        rights[detected],
    )

    lefts, rights = ext.split_peaks(prominence, lefts, rights, prominence_required)
    regions = np.stack((lefts, rights), axis=1)

    indicies = regions.ravel()
    if indicies.size > 0 and indicies[-1] == y.size:
        indicies = indicies[:-1]
    # Get number above limit in each region
    num_detections = np.add.reduceat(above, indicies)[::2]
    # Remove regions without minimum_size values above detection limit
    regions = regions[num_detections >= points_required]

    indicies = regions.ravel()
    if indicies.size > 0 and indicies[-1] == y.size:
        indicies = indicies[:-1]

    # Sum regions
    if integrate:
        base = y - limit_accumulation
        base[base < 0.0] = 0.0
        sums = np.add.reduceat(base, indicies)[::2]
    else:
        sums = np.add.reduceat(y, indicies)[::2]

    return sums, regions


def background_mask(regions: np.ndarray, size: int) -> np.ndarray:
    """Create a mask to select background data using ``regions``.

    Args:
        regions: regions from ``accumulate_detections``
        size: size of underlying array, e.g. signals

    Returns:
        mask of signal background
    """
    labels = ext.label_regions(regions, size)
    return labels == 0


def combine_regions(regions: list[np.ndarray], overlap: int) -> np.ndarray:
    """Combine regions with an optional allowed overlap.

    Regions in ``regions`` that overlap by more than ``overlap`` are merged.

    Args:
        regions: list of regions from ``accumulate_detections``
        overlap: maximum allowed overlap

    Returns:
        merged regions
    """
    return ext.combine_regions(regions, overlap)


def detection_maxima(y: np.ndarray, regions: np.ndarray) -> np.ndarray:
    """Calculates the maxima of each region.

    Does not work with overlapping regions.

    Args:
        y: array
        regions: regions from ``accumulate_detections``

    Returns:
        idx of maxima
    """

    idx = ext.maxima_between(y, regions)
    return idx
