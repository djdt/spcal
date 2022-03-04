import numpy as np
import spcal.poisson

stapleton_test_sd = [
    5.411,
    10.063,
    11.991,
    13.469,
    14.716,
    15.814,
    16.807,
    17.720,
    18.570,
    19.368,
    20.123,
    20.841,
    21.527,
    22.185,
    22.819,
    23.430,
    24.020,
    24.593,
    25.149,
    25.690,
    26.217,
]


def test_poisson_stapleton():
    # Table 20.3
    for i in range(len(stapleton_test_sd)):
        _, sd = spcal.poisson.stapleton_approximation(i)
        assert np.isclose(sd, stapleton_test_sd[i], atol=1e-3)

    # Example 20.10
    sc, sd = spcal.poisson.stapleton_approximation(4)
    assert np.isclose(sc, 6.23, atol=1e-2)

    # Equation 20.75
    sc, sd = spcal.poisson.stapleton_approximation(1.23)
    assert np.isclose(5.41 + 4.65 * np.sqrt(1.23), sd, atol=1e-2)

def test_poisson():
    # Test close to the values in Currie 2007
    yc, yd = spcal.poisson_limits(4.0, epsilon=0.0)
    assert all(np.isclose([yc, yd], [2.326 * 2.0, 2.71 + 4.65 * 2.0], atol=1e-3))


def test_poisson_alpha_zero():
    for alpha in np.arange(0.05, 0.51, 0.05):
        yc, yd = spcal.poisson_limits(4.0, alpha=alpha, beta=0.50, epsilon=0.0)
        print(alpha, yc, yd)
