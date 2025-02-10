"""Functions for detecting and classifying particles."""

import numpy as np

from spcal.lib.spcalext import detection as ext


def accumulate_detections(
    y: np.ndarray,
    limit_accumulation: float | np.ndarray,
    limit_detection: float | np.ndarray,
    points_required: int = 1,
    promience_required: float | np.ndarray | None = None,
    integrate: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns an array of accumulated detections.

    Peak prominence is calculated for all points above the ``limit_detection``,
    with widths bound by the ``limit_accumulation``.
    Detections are peaks with a prominence greater than ``promience_required`` and at
    least ``points_required`` points above the ``limit_detection``.

    Args:
        y: array
        limit_accumulation: minimum accumulation value(s)
        limit_detection: minimum detection value(s)
        points_required: no. points > limit_detection to be detected
        prominence_required: minimum prominence for peaks
        integrate: integrate, otherwise sum

    Returns:
        summed detection regions
        labels of regions
        regions [starts, ends]
    """

    def local_maxima(x: np.ndarray):
        return np.logical_and(
            np.r_[False, x[1:] >= x[:-1]], np.r_[x[:-1] >= x[1:], False]
        )

    if np.any(limit_accumulation > limit_detection):
        raise ValueError("accumulate_detections: limit_accumulation > limit_detection.")
    if points_required < 1:
        raise ValueError("accumulate_detections: minimum size must be >= 1")
    if promience_required is None:
        promience_required = limit_detection

    # todo: see if smoothing required
    # todo: perfoemance of possible_detections, detections and sum
    # psf = normal.pdf(np.linspace(-2, 2, 5))
    # ysm = np.convolve(y, psf / psf.sum(), mode="same")

    possible_detections = np.flatnonzero(
        np.logical_and(y > limit_detection, local_maxima(y))
    )

    prominence, lefts, rights = ext.peak_prominence(
        y, possible_detections, min_value=limit_accumulation
    )

    # if promience_required is array, use mid points
    if isinstance(promience_required, np.ndarray):
        detected = prominence >= promience_required[lefts + (rights - lefts) // 2]
    else:
        detected = prominence >= promience_required
    prominence, lefts, rights = (
        prominence[detected],
        lefts[detected],
        rights[detected],
    )

    # fix any overlapped peaks, prefering most prominent
    right_larger = prominence[:-1] < prominence[1:]
    lefts[1:][right_larger] = np.maximum(
        lefts[1:][right_larger], rights[:-1][right_larger]
    )
    rights[:-1][~right_larger] = np.minimum(
        lefts[1:][~right_larger], rights[:-1][~right_larger]
    )

    regions = np.stack((lefts, rights), axis=1)

    indicies = regions.ravel()
    if indicies.size > 0 and indicies[-1] == y.size:
        indicies = indicies[:-1]

    detections = np.add.reduceat(y > limit_detection, indicies)[::2]
    regions = regions[detections >= points_required]

    indicies = regions.ravel()
    if indicies.size > 0 and indicies[-1] == y.size:
        indicies = indicies[:-1]

    # Sum regions
    if integrate:
        sums = np.add.reduceat(np.clip(y - limit_accumulation, 0.0, None), indicies)[
            ::2
        ]
    else:
        sums = np.add.reduceat(y, indicies)[::2]

    # Create a label array of detections
    labels = ext.label_regions(regions, y.size)

    return sums, labels, regions


def combine_detections(
    sums: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    regions: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the relative fraction of each element in each detection.
    Recalculates the start and end point of each peak from *all* element data.
    Regions that overlap will be combined into a single region. Fractions are calculated
    as the sum of all regions contained within each of the reclaculated regions.
    Each argument must have the same dictionaty keys.

    Args:
        sums: dict of detection counts, sizes, mass, ...
        labels: dict of labels from `accumulate_detections`
        regions: dict of regions from `accumulate_detections`

    Returns:
        dict of total sum per peak
        combined labels
        combined regions

    """
    if not all(k in regions.keys() for k in sums.keys()):  # pragma: no cover
        raise ValueError(
            "detection_element_combined: labels and regions must have all of sums keys."
        )
    names = list(sums.keys())

    # Get regions from all elements
    all_regions = ext.combine_regions(list(regions.values()), 0)

    any_label = ext.label_regions(all_regions, next(iter(labels.values())).size)

    # Init to zero, summed later
    combined = np.zeros(
        all_regions.shape[0], dtype=[(name, np.float64) for name in sums]
    )
    region_used = np.zeros(all_regions.shape[0])
    for name in names:
        idx = np.searchsorted(all_regions[:, 0], regions[name][:, 0], side="right") - 1
        np.add.at(region_used, idx, 1)
        np.add.at(combined[name], idx, sums[name])

    return combined, any_label, all_regions


def detection_maxima(y: np.ndarray, regions: np.ndarray) -> np.ndarray:
    """Calculates the maxima of each region.

    Does not work with overlapping regions.

    Args:
        y: array
        regions: regions from `accumulate_detections`

    Returns:
        idx of maxima
    """

    idx = ext.maxima(y, regions)
    return idx
