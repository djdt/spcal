import numpy as np
import matplotlib.pyplot as plt
from math import factorial

from spcal import poisson


def probability(fn, u: float, alpha: float = 0.05, lim: int = 100) -> float:
    if u <= 0.0:
        return 0.0

    u = float(u)

    N = np.arange(0, lim)
    yc = np.array([int(fn(n, alpha=alpha)[0]) + n for n in N])

    K = [u**k / factorial(k) for k in np.arange(0, yc.max())]

    return 1.0 - np.exp(-u * 2.0) * np.sum(
        [(u**n / factorial(n)) * np.sum(K[: yc[i] + 1]) for i, n in enumerate(N)]
    )


x = np.arange(0.0, 50.0, 0.25)

fig, axes = plt.subplots(3, 1, sharex=True)

axes[0].plot(
    x, [probability(poisson.formula_a, u, alpha=0.01) for u in x], label="Formula A"
)
axes[0].plot(
    x, [probability(poisson.formula_c, u, alpha=0.01) for u in x], label="Formula C"
)
axes[0].plot(
    x,
    [probability(poisson.stapleton_approximation, u, alpha=0.01) for u in x],
    label="Stapleton",
)
axes[0].axhline(0.01, color="black", ls="--")

axes[1].plot(x, [probability(poisson.formula_a, u, alpha=0.05) for u in x])
axes[1].plot(x, [probability(poisson.formula_c, u, alpha=0.05) for u in x])
axes[1].plot(
    x, [probability(poisson.stapleton_approximation, u, alpha=0.05) for u in x]
)
axes[1].axhline(0.05, color="black", ls="--")

axes[2].plot(x, [probability(poisson.formula_a, u, alpha=0.1) for u in x])
axes[2].plot(x, [probability(poisson.formula_c, u, alpha=0.1) for u in x])
axes[2].plot(x, [probability(poisson.stapleton_approximation, u, alpha=0.1) for u in x])
axes[2].axhline(0.1, color="black", ls="--")

axes[0].legend()
axes[0].set_title("α=0.01")
axes[1].set_title("α=0.05")
axes[2].set_title("α=0.1")

axes[1].set_ylabel("P")
axes[2].set_xlabel("Mean Background")

plt.tight_layout()
plt.show()
