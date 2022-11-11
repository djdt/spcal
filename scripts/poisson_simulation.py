from typing import Tuple
import numpy as np

from spcal import poisson

import matplotlib.pyplot as plt

gen = np.random.default_rng()


def false_positive_rates(x: np.ndarray, alpha: float) -> Tuple[float, float, float]:
    u = x.mean()
    sc1, _ = poisson.formula_a(u, alpha=alpha)
    sc2, _ = poisson.formula_c(u, alpha=alpha)
    sc3, _ = poisson.stapleton_approximation(u, alpha=alpha)

    # return sc1, sc2, sc3
    return tuple(np.count_nonzero(x >= int(sc + u)) / x.size for sc in [sc1, sc2, sc3])


def rates_for_means(means: np.ndarray) -> np.ndarray:
    alphas = np.array([0.01, 0.05, 0.1])
    rates = np.empty(
        (means.size, alphas.size),
        dtype=[("A", float), ("C", float), ("Stapleton", float)],
    )
    for i, u in enumerate(means):
        x = gen.poisson(lam=u, size=100000)
        for j, alpha in enumerate(alphas):
            scs = false_positive_rates(x.astype(float), alpha=alpha)
            rates[i, j] = false_positive_rates(x.astype(float), alpha=alpha)

    return rates


rates = rates_for_means(np.arange(1, 101))


fig, axes = plt.subplots(3, 1)


axes[0].plot(rates[:, 0]["A"], label="Formula A")
axes[0].plot(rates[:, 0]["C"], label="Formula C")
axes[0].plot(rates[:, 0]["Stapleton"], label="Stapleton")

axes[1].plot(rates[:, 1]["A"], label="Formula A")
axes[1].plot(rates[:, 1]["C"], label="Formula C")
axes[1].plot(rates[:, 1]["Stapleton"], label="Stapleton")

axes[2].plot(rates[:, 2]["A"], label="Formula A")
axes[2].plot(rates[:, 2]["C"], label="Formula C")
axes[2].plot(rates[:, 2]["Stapleton"], label="Stapleton")
plt.show()
