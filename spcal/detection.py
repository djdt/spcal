"""Functions for detecting and classifying particles."""
import numpy as np
import logging

from typing import Dict, Tuple

logger = logging.getLogger(__name__)


def accumulate_detections(
    y: np.ndarray,
    limit_accumulation: float | np.ndarray,
    limit_detection: float | np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns an array of accumulated detections.

    Contiguous regions above `limit_accumulation` that contain at least one value above
    `limit_detection` are summed.

    Args:
        y: array
        limit_detection: value(s) for detection of region
        limit_accumulation: minimum accumulation value(s)

    Returns:
        summed detection regions
        labels of regions
        regions [starts, ends]
    """
    if np.any(limit_detection < limit_accumulation):
        raise ValueError("limit_detection must be greater than limit_accumulation.")
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

    # Get maximum values in each region
    detections = np.logical_or.reduceat(y > limit_detection, regions.ravel())[::2]
    # Remove regions without a max value above detection limit
    regions = regions[detections]
    # Sum regions
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

    Args:
        y: array
        regions: regions from `accumulate_detections`

    Returns:
        idx of maxima
    """

    def argmax_reduceat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        idx = np.zeros(a.size, dtype=int)
        idx[b[1:]] = 1
        shift = (a.max() + 1) * np.cumsum(idx)
        sortidx = np.argsort(a + shift)
        return sortidx[np.append(b[1:], a.size) - 1] - b

    return argmax_reduceat(y, regions.ravel())[::2] + regions[:, 0]


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
    if not all([k in regions.keys() for k in sums.keys()]):
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


def fraction_components(
    fractions: np.ndarray,
    bins: np.ndarray | None = None,
    combine_similar: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Binned ratios for element fractions.
    Calculates the mean value and count of peaks with a combination of elements separated by `bins`.
    This can be used to find the different elemental composition of NPs in a sample.

    Args:
        fractions: ratios returned by `detection_element_fractions`
        bins: bins for each element fraction, defaults to [0.0, 0.1, ... 0.9]
        combine_similar: combine compositions with a difference less than half the bin width

    Returns:
        mean of each combination
        count of each combination"""
    if bins is None:
        bins = np.linspace(0, 0.9, 10)

    # Generate indicies from histogram of each element
    hist = np.stack(
        [np.digitize(fractions[name], bins=bins) for name in fractions.dtype.names],
        axis=1,
    )

    # Unique combinations across all histogram indicies
    _, idx, counts = np.unique(hist, axis=0, return_inverse=True, return_counts=True)

    # Calculate the mean value for each unique combination
    means = np.empty(counts.size, dtype=fractions.dtype)
    for name in fractions.dtype.names:
        means[name] = np.bincount(idx, fractions[name]) / counts

    idx = np.argsort(counts)[::-1]
    means = means[idx]
    counts = counts[idx]

    def rec_similar(rec: np.ndarray, x: np.ndarray, diff: float) -> np.ndarray:
        return np.all(
            [np.abs(rec[name] - x[name]) < diff for name in x.dtype.names],
            axis=0,
        )

    def rec_mean(rec: np.ndarray) -> np.ndarray:
        return np.array(
            tuple(np.mean(rec[name]) for name in rec.dtype.names), dtype=rec.dtype
        )

    _iter = 0
    if combine_similar:
        # diff = np.sqrt(len(fractions.dtype.names) * (bins[1] - bins[0]) ** 2)
        # Todo Come up with better distance, not flat in all dims
        diff = bins[1] - bins[0]
        combined_means = []
        combined_counts = []
        while means.size > 0 and idx.size > 0:
            idx = np.flatnonzero(rec_similar(means, means[0], diff))
            combined_means.append(rec_mean(means[idx]))
            combined_counts.append(np.sum(counts[idx]))
            means = np.delete(means, idx)
            counts = np.delete(counts, idx)

            _iter += 1
            if _iter > 100:
                logger.warn("Maximum iter reached for combine_similar.")
                break
        means, counts = np.array(combined_means, dtype=means.dtype), np.array(
            combined_counts
        )

    return means, counts
