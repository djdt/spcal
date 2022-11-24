import numpy as np

from spcal import fit


def test_neadler_mean():
    def gradient(x: np.ndarray, y: np.ndarray, a: float, b: float) -> float:
        return np.sum(np.abs((x * a + np.log(b)) - y))

    x = np.arange(100.0)
    y = x * 3.0 + np.log(4.0)

    simplex = np.array([[1.0, 1.0], [10.0, 1.0], [1.0, 10.0]])

    args = fit.nelder_mead(gradient, x, y, simplex)

    assert np.isclose(args[0], 3.0)
    assert np.isclose(args[1], 4.0, atol=1e-3)
