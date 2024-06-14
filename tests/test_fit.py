import numpy as np

from spcal.fit import bfgs

# def test_bfgs():
#     def gradient(x: np.ndarray, y: np.ndarray, a: float, b: float) -> float:
#         return np.sum(np.abs((x * a + np.log(b)) - y))
#
#     x = np.arange(100.0)
#     y = x * 3.0 + np.log(4.0)
#
#     simplex = np.array([[1.0, 1.0], [10.0, 1.0], [1.0, 10.0]])
#
#     args = fit.nelder_mead(gradient, x, y, simplex)
#
#     assert np.isclose(args[0], 3.0)
#     assert np.isclose(args[1], 4.0, atol=1e-3)


def rosenbrock(x: float, y: float) -> float:
    return (1.0 - x) ** 2 + 100.0 * (y - x**2) ** 2


def rosenbrock_grad(x: float, y: float) -> np.ndarray:
    return np.array(
        [2.0 * (200.0 * x**3 - 200.0 * x * y + x - 1.0), 200.0 * (y - x**2)]
    )


def test_bfgs():
    def fn(p: np.ndarray) -> float:
        return rosenbrock(p[0], p[1])

    def grad_fn(p: np.ndarray) -> float:
        return rosenbrock_grad(p[0], p[1])

    x0 = np.array([-1.0, -1.0])
    opt = bfgs(fn, x0, grad_fn=grad_fn, max_iter=1000, eps=1e-3)
    print(opt)


test_bfgs()
