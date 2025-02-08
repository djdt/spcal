"""Functions for detecting and classifying particles."""

import numpy as np

from spcal.dists import normal
from spcal.lib.spcalext import combine_regions, label_regions, maxima, peak_prominence


def _contiguous_regions(x: np.ndarray, limit: float | np.ndarray) -> np.ndarray:
    """Returns start and end points of regions in x that are greater than limit.
    Indexs to the start point and point after region.

    Args:
        x: array
        limit: minimum value in regions

    Returns:
        regions [start, end]
    """

    # Get start and end positions of regions above accumulation limit
    diff = np.diff((x > limit).astype(np.int8), prepend=0)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    # Stack into pairs of start, end. If no final end position set it as end of array.
    if starts.size != ends.size:
        # -1 for reduceat
        ends = np.concatenate((ends, [diff.size]))  # type: ignore
    return np.stack((starts, ends), axis=1)


def accumulate_detections(
    y: np.ndarray,
    limit_accumulation: float | np.ndarray,
    limit_detection: float | np.ndarray,
    points_required: int = 1,
    integrate: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns an array of accumulated detections.

    Contiguous regions above ``limit_accumulation`` that contain at least
    ``points_required`` values above ``limit_detection`` are summed or integrated
    (sum - ``limit_accumulation``).

    Args:
        y: array
        limit_accumulation: minimum accumulation value(s)
        limit_detection: minimum detection value(s)
        points_required: no. points > limit_detection to be detected
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

    # todo: see if smoothing required
    # psf = normal.pdf(np.linspace(-2, 2, 5))
    # ysm = np.convolve(y, psf / psf.sum(), mode="same")

    possible_detections = np.flatnonzero(
        np.logical_and(y > limit_detection, local_maxima(y))
    )
    prominence, lefts, rights = peak_prominence(
        y, possible_detections, limit_accumulation
    )

    detected = prominence > limit_detection * 0.9

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
        sums = np.add.reduceat(y - limit_accumulation, indicies)[::2]
    else:
        sums = np.add.reduceat(y, indicies)[::2]

    # Create a label array of detections
    labels = label_regions(regions, y.size)

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
    import time

    if not all(k in regions.keys() for k in sums.keys()):  # pragma: no cover
        raise ValueError(
            "detection_element_combined: labels and regions must have all of sums keys."
        )
    names = list(sums.keys())

    # todo: find a way to preserve touching regions

    # Get regions from all elements
    # Some regions may overlap, these will be combined
    t0 = time.time()
    any_label = np.zeros(labels[names[0]].size, dtype=np.int8)
    for name in names:
        any_label[labels[name] > 0] = 1

    all_regions = _contiguous_regions(any_label, 0)
    t1 = time.time()
    print(list(regions.values())[0])
    all_regions = combine_regions(list(regions.values()))
    # current_pos = np.zeros(len(regions), dtype=int)
    # new = []
    # iter = 0
    # while True:
    #     iter += 1
    #     if np.all(current_pos == [x.shape[0] for x in regions.values()]):
    #         break
    #
    #     left, right = np.inf, 0
    #     leftmost = 0
    #     for i, regs in enumerate(regions.values()):
    #         if current_pos[i] == regs.shape[0]:
    #             continue
    #         # left most current region
    #         s, e = regs[current_pos[i]]
    #         if s < left:
    #             left = s
    #             right = e
    #             leftmost = i
    #
    #     current_pos[leftmost] += 1
    #     changed = True
    #     while changed:
    #         changed = False
    #         # print("start while", right)
    #         for i, regs in enumerate(regions.values()):
    #             # print("loop", i)
    #             if current_pos[i] == regs.shape[0]:
    #                 continue
    #             # new_right = right
    #             s, e = regs[current_pos[i]]
    #             if s < right and e >= right:
    #                 right = max(right, e)
    #                 current_pos[i] += 1
    #                 changed = True
    #                 print("\t\t changed ")
    #             elif e < right:
    #                 current_pos[i] += 1
    #
    #     new.append((left, right))
    #     all_regions = np.array(new)
    print(all_regions, any_label.size)
    t2 = time.time()
    print("old", t1 - t0)
    print("new", t2 - t1)

    any_label = label_regions(all_regions, any_label.size)

    # Init to zero, summed later
    combined = np.zeros(
        all_regions.shape[0], dtype=[(name, np.float64) for name in sums]
    )
    for name in names:
        idx = np.searchsorted(all_regions[:, 0], regions[name][:, 0], side="right") - 1
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

    idx = maxima(y, regions)
    return idx
