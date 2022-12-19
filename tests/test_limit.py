import numpy as np

from spcal.limit import SPCalLimit

x = np.random.poisson(lam=50.0, size=1000)


# def test_limit_errors():
#     with pytest.raises(ValueError):
#         calc.calculate_limits(np.array([]), "Automatic")


def test_limit_from_poisson():
    lim = SPCalLimit.fromPoisson(x, alpha=0.05,  beta=0.05)  # ld ~= 87
    assert lim.name == "Poisson"
    assert lim.params == {"alpha": 0.05, "beta": 0.05}
    assert lim.limit_of_criticality, lim.limit_of_detection == poisson.formula_c(
        np.mean(x)
    ) + np.mean(x)

def test_limit_from_gaussian():
    lim = SPCalLimit.fromGaussian(x, sigma=5.0)  # ld ~= 87
    assert lim.name == "Gaussian"
    assert lim.params == {"sigma": 5.0}
    assert lim.limit_of_criticality, lim.limit_of_detection == np.mean(x) + 5.0 * np.std(x)

    lim = SPCalLimit.fromGaussian(x, sigma=5.0, use_median=True)  # ld ~= 87
    assert lim.name == "Gaussian Median"
    assert lim.params == {"sigma": 5.0}
    assert lim.limit_of_criticality, lim.limit_of_detection == np.median(x) + 5.0 * np.std(x)

def test_limit_from_highest():
    lim = SPCalLimit.fromHighest(x, sigma=5.0)
    assert lim.name == "Poisson"

def test_limit_windowed():
    lim = SPCalLimit.fromPoisson(x, window_size=3)
    assert lim.window_size == 3
    assert lim.limit_of_detection.size == x.size

def test_limit_from_best():  # Better way for normality check?
    for lam in np.linspace(1.0, 100.0, 25):
        x = np.random.poisson(size=1000, lam=lam)
        lim = SPCalLimit.fromBest(x, sigma=3.0, alpha=0.05, beta=0.05)
        assert lim.name == ("Poisson" if lam < 50.0 else "Gaussian")
