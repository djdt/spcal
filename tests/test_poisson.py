import numpy as np
import spcal.poisson

estimated_sd_values = [
    # (mean, FA, Stapleton)
    (0, 2.706, 5.411),
    (1, 7.358, 10.063),
    (2, 9.285, 11.991),
    (3, 10.764, 13.469),
    (4, 12.010, 14.716),
    (5, 13.109, 15.814),
    (6, 14.101, 16.807),
    (7, 15.015, 17.720),
    (8, 15.864, 18.570),
    (9, 16.663, 19.368),
    (10, 17.418, 20.123),
    (11, 18.136, 20.841),
    (12, 18.822, 21.527),
    (13, 19.480, 22.185),
    (14, 20.113, 22.819),
    (15, 20.724, 23.430),
    (16, 21.315, 24.020),
    (17, 21.888, 24.593),
    (18, 22.444, 25.149),
    (19, 22.985, 25.690),
    (20, 23.511, 26.217),
]


def test_poisson_formula_a():
    # Table 20.3
    for b, est, _ in estimated_sd_values:
        _, sd = spcal.poisson.formula_a(b, alpha=0.05, beta=0.05)
        assert np.isclose(sd, est, atol=1e-3)

    # Example 20.10
    sc, _ = spcal.poisson.formula_a(4, alpha=0.05, beta=0.05)
    assert np.isclose(sc, 4.65, atol=1e-2)

    # When α = 0.05, sc = 2.33 sqrt(b)
    for b in np.arange(0, 5, 0.5):
        sc, _ = spcal.poisson.formula_a(b, alpha=0.05)
        assert np.isclose(sc, 2.33 * np.sqrt(b), atol=1e-2)


def test_poisson_formula_a_error_rates():
    pass

def test_poisson_stapleton():
    # Table 20.3
    for b, _, est in estimated_sd_values:
        _, sd = spcal.poisson.stapleton_approximation(b, alpha=0.05, beta=0.05)
        assert np.isclose(sd, est, atol=1e-3)

    # Example 20.10
    sc, _ = spcal.poisson.stapleton_approximation(4, alpha=0.05, beta=0.05)
    assert np.isclose(sc, 6.23, atol=1e-2)

    # Equation 20.75
    for b in np.arange(0, 5, 0.5):
        _, sd = spcal.poisson.stapleton_approximation(b, alpha=0.05, beta=0.05)
        assert np.isclose(5.41 + 4.65 * np.sqrt(b), sd, atol=1e-2)


def test_poisson_stapleton_error_rates():
    pass
