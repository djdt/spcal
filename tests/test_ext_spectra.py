import numpy as np
from spcal.lib.spcalext import spectra


def test_spectra():
    x = np.random.random((100, 10))
    regions = np.array([[10, 20], [30, 50], [70, 90]])

    y = spectra.spectra(x, regions)
    assert y.shape == (3, 10)
    assert np.allclose(y[0, :], np.mean(x[10:20], axis=0))
    assert np.allclose(y[1, :], np.mean(x[30:50], axis=0))
    assert np.allclose(y[2, :], np.mean(x[70:90], axis=0))

    y = spectra.spectra(x, regions, False)
    assert y.shape == (3, 10)
    assert np.allclose(y[0, :], np.sum(x[10:20], axis=0))
    assert np.allclose(y[1, :], np.sum(x[30:50], axis=0))
    assert np.allclose(y[2, :], np.sum(x[70:90], axis=0))
