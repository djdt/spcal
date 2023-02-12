"""Functions for detecting and classifying particles."""
from typing import Dict, Tuple

import numpy as np

from spcal.lib.spcalext import maxima


def accumulate_detections(
    y: np.ndarray,
    limit_accumulation: float | np.ndarray,
    limit_detection: float | np.ndarray,
    integrate: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns an array of accumulated detections.

    Contiguous regions above `limit_accumulation` that contain at least one value above
    `limit_detection` are summed or integrated (sum - `limit_accumulation`).

    Args:
        y: array
        limit_accumulation: minimum accumulation value(s)
        limit_detection: minimum detection value(s)
        integrate: integrate, otherwise sum

    Returns:
        summed detection regions
        labels of regions
        regions [starts, ends]
    """
    # Get start and end positions of regions above accumulation limit
    diff = np.diff((y > limit_accumulation).astype(np.int8), prepend=0)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    # Stack into pairs of start, end. If no final end position set it as end of array.
    end_point_added = False
    if starts.size != ends.size:
        # -1 for reduceat
        ends = np.concatenate((ends, [diff.size - 1]))  # type: ignore
        end_point_added = True
    regions = np.stack((starts, ends), axis=1)

    # Get maximum in each region
    detections = np.logical_or.reduceat(y > limit_detection, regions.ravel())[::2]
    # Remove regions without a max value above detection limit
    regions = regions[detections]
    # Sum regions
    if integrate:
        sums = np.add.reduceat(y - limit_accumulation, regions.ravel())[::2]
    else:
        sums = np.add.reduceat(y, regions.ravel())[::2]

    # Create a label array of detections
    labels = np.zeros(y.size, dtype=np.int16)
    ix = np.arange(1, regions.shape[0] + 1)
    # Set start, end pairs to +i, -i
    labels[regions[:, 0]] = ix
    if end_point_added:
        labels[regions[:-1, 1]] = -ix[:-1]
    else:
        labels[regions[:, 1]] = -ix
    # Cumsum to label
    labels = np.cumsum(labels)

    return sums, labels, regions


def detection_maxima(y: np.ndarray, regions: np.ndarray) -> np.ndarray:
    """Calculates the maxima of each region.
    Does not work with overlapping regions.

    Args:
        y: array
        regions: regions from `accumulate_detections`

    Returns:
        idx of maxima
    """

    # def maxima(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    #     idx = np.zeros(a.size, dtype=int)
    #     idx[b[1:]] = 1
    #     shift = (a.max() + 1) * np.cumsum(idx)
    #     sortidx = np.argsort(a + shift)
    #     return sortidx[np.append(b[1:], a.size) - 1] - b
    idx = maxima(y, regions)
    return idx


def combine_detections(
    sums: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray],
    regions: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    """
    if not all(k in regions.keys() for k in sums.keys()):  # pragma: no cover
        raise ValueError(
            "detection_element_combined: labels and regions must have all of sums keys."
        )
    names = list(sums.keys())

    # Get regions from all elements
    # Some regions may overlap, these will be combined
    any_label = np.zeros(labels[names[0]].size, dtype=np.int16)
    for name in names:
        any_label[labels[name] > 0] = 1

    # Caclulate start and end points of each region
    diff = np.diff(any_label, prepend=0)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)

    # Stack into pairs of start, end. If no final end position set it as end of array.
    if starts.size != ends.size:
        ends = np.concatenate((ends, [diff.size - 1]))  # type: ignore
        end_point_added = True
    else:
        end_point_added = False
    all_regions = np.stack((starts, ends), axis=1)

    ix = np.arange(1, all_regions.shape[0] + 1)
    # Set start, end pairs to +i, -i
    any_label[:] = 0
    any_label[all_regions[:, 0]] = ix
    if end_point_added:
        any_label[all_regions[:-1, 1]] = -ix[:-1]
    else:
        any_label[all_regions[:, 1]] = -ix
    # Cumsum to label
    any_label = np.cumsum(any_label)
    # Init empty
    combined = np.empty(
        all_regions.shape[0], dtype=[(name, np.float64) for name in sums]
    )
    for name in names:
        # Positions in name's region that corresponds to the combined regions
        idx = (regions[name][:, 0] >= all_regions[:, 0, None]) & (
            regions[name][:, 1] <= all_regions[:, 1, None]
        )
        combined[name] = np.sum(np.where(idx, sums[name], 0), axis=1)

    return combined, any_label, all_regions
