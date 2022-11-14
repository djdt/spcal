import numpy as np

import spcal.poisson

estimated_sd_values = [
    # (mean, FA, True, FC, True, Stapleton, True)
    (0, 2.706, 2.996, 7.083, 6.296, 5.411, 6.296),
    (1, 7.358, 8.351, 9.660, 10.095, 10.063, 10.095),
    (2, 9.285, 10.344, 11.355, 12.010, 11.991, 12.010),
    (3, 10.764, 11.793, 12.719, 13.551, 13.469, 13.551),
    (4, 12.010, 13.021, 13.894, 14.826, 14.716, 14.826),
    (5, 13.109, 14.091, 14.942, 15.930, 15.814, 15.930),
    (6, 14.101, 15.076, 15.897, 16.902, 16.807, 16.902),
    (7, 15.015, 16.028, 16.780, 17.785, 17.720, 17.785),
    (8, 15.864, 16.945, 17.605, 18.614, 18.570, 18.614),
    (9, 16.663, 17.804, 18.383, 19.406, 19.368, 19.406),
    (10, 17.418, 18.595, 19.120, 20.170, 20.123, 20.170),
    (11, 18.136, 19.324, 19.823, 20.903, 20.841, 20.903),
    (12, 18.822, 20.002, 20.496, 21.602, 21.527, 21.602),
    (13, 19.480, 20.642, 21.142, 22.267, 22.185, 22.267),
    (14, 20.113, 21.257, 21.764, 22.900, 22.819, 22.900),
    (15, 20.724, 21.854, 22.366, 23.506, 23.430, 23.506),
    (16, 21.315, 22.438, 22.948, 24.091, 24.020, 24.091),
    (17, 21.888, 23.010, 23.513, 24.657, 24.593, 24.657),
    (18, 22.444, 23.569, 24.062, 25.206, 25.149, 25.206),
    (19, 22.985, 24.116, 24.596, 25.738, 25.690, 25.738),
    (20, 23.511, 24.649, 25.116, 26.252, 26.217, 26.252),
]


def test_poisson_currie():
    # Table II Currie 1968
    sc, sd = spcal.poisson.currie(0.0, alpha=0.05, beta=0.05, epsilon=0.0, eta=2.0)
    assert np.isclose(sc, 0.0)
    assert np.isclose(sd, 2.71, atol=1e-2)
    # Table II Currie 1968
    sc, sd = spcal.poisson.currie(1.0, alpha=0.05, beta=0.05, epsilon=0.0, eta=2.0)
    assert np.isclose(sc, 2.33, atol=1e-2)
    assert np.isclose(sd, 2.71 + 4.65, atol=1e-2)
    # Example p. 592
    sc, sd = spcal.poisson.currie(308.0, alpha=0.05, beta=0.05, epsilon=0.0, eta=2.0)
    assert np.isclose(sc, 40.8, atol=1e-1)


def test_poisson_sc_formula_a():
    # Example 20.10
    sc, _ = spcal.poisson.formula_a(4, alpha=0.05, beta=0.05)
    assert np.isclose(sc, 4.65, atol=1e-2)

    # Example 20.11
    sc, _ = spcal.poisson.formula_a(
        108, alpha=0.05, beta=0.05, t_sample=3000, t_blank=6000
    )
    assert np.isclose(sc, 14.8, atol=1e-2)

    # When α = 0.05, sc = 2.33 sqrt(b)
    for b in np.arange(0, 5, 0.5):
        sc, _ = spcal.poisson.formula_a(b, alpha=0.05)
        assert np.isclose(sc, 2.33 * np.sqrt(b), atol=1e-2)


def test_poisson_sc_formula_c():
    # Example 20.10
    sc, _ = spcal.poisson.formula_c(4, alpha=0.05, beta=0.05)
    assert np.isclose(sc, 6.20, atol=1e-2)

    # Example 20.11
    sc, _ = spcal.poisson.formula_c(
        108, alpha=0.05, beta=0.05, t_sample=3000, t_blank=6000
    )
    assert np.isclose(sc, 15.5, atol=1e-1)


def test_poisson_sc_stapleton():
    # Example 20.10
    sc, _ = spcal.poisson.stapleton_approximation(4, alpha=0.05, beta=0.05)
    assert np.isclose(sc, 6.23, atol=1e-2)

    # Example 20.11
    sc, _ = spcal.poisson.stapleton_approximation(
        108, alpha=0.05, beta=0.05, t_sample=3000, t_blank=6000
    )
    assert np.isclose(sc, 15.6, atol=1e-1)

    # Equation 20.75
    for b in np.arange(0, 5, 0.5):
        _, sd = spcal.poisson.stapleton_approximation(b, alpha=0.05, beta=0.05)
        assert np.isclose(5.41 + 4.65 * np.sqrt(b), sd, atol=1e-2)


def test_poisson_sd_estimated():
    # Table 20.3
    for ub, fa, _, fc, _, stapleton, _ in estimated_sd_values:
        assert np.isclose(fa, spcal.poisson.formula_a(ub)[1], atol=1e-3)
        assert np.isclose(fc, spcal.poisson.formula_c(ub)[1], atol=1e-3)
        assert np.isclose(
            stapleton, spcal.poisson.stapleton_approximation(ub)[1], atol=1e-3
        )
