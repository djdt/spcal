import numpy as np

from spcal import poisson, accumulate_detections

import matplotlib.pyplot as plt


def otsu(x: np.ndarray, remove_nan: bool = False) -> float:
    """Calculates the otsu threshold.

    The Otsu threshold minimises intra-class variance for a two class system.
    If `remove_nan` then all nans are removed before computation.

    Args:
        x: array
        remove_nan: remove nan values

    See Also:
        :func:`skimage.filters.threshold_otsu`
    """
    if remove_nan:
        x = x[~np.isnan(x)]

    hist, bin_edges = np.histogram(x, bins=256)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0

    w1 = np.cumsum(hist)
    w2 = np.cumsum(hist[::-1])[::-1]

    u1 = np.cumsum(hist * bin_centers) / w1
    u2 = (np.cumsum((hist * bin_centers)[::-1]) / w2[::-1])[::-1]

    i = np.argmax(w1[:-1] * w2[1:] * (u1[:-1] - u2[1:]) ** 2)
    return bin_centers[i]

gen = np.random.default_rng()

fig, ax = plt.subplots()

for noise_level in [1.0, 5.0, 10.0]:
    bg = np.random.poisson(noise_level, 1000000)

    particles = np.geomspace(10, 100000, num=50)

    d1 = []
    d2 = []

    for p in particles:
        x = bg.copy()
        n = np.random.poisson(100.0, int(p))
        x[np.random.choice(x.size, size=n.size, replace=False)] = n

        ub = x.mean()
        ob = x[x<otsu(x)].mean()
        lc, ld = poisson.formula_c(ub, 0.05)
        lc2, ld2 = poisson.formula_c(x[x<otsu(x)].mean(), 0.05)

        d1.append(ub)
        d2.append(ob)
        # d1.append(accumulate_detections(x, lc, ld)[0].size)
        # d2.append(accumulate_detections(x, lc, ld)[0].size)

    ax.plot(particles, d1)
    ax.plot(particles, d2)

plt.show()
