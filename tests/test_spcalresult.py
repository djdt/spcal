import numpy as np
import pytest
from numpy.lib import stride_tricks

from spcal.limit import SPCalLimit
from spcal.particle import particle_size
from spcal.result import SPCalResult

signal = np.random.random(50)
signal[5:10] = 10.0
signal[15:18] = 20.0
signal[34:38] = 30.0
detections = np.array([10 * 5, 20 * 3, 30 * 4, 0])
labels = np.zeros(50)
labels[signal > 5] = 1
labels[45:46] = 1


def test_spcalresult():
    result = SPCalResult(
        "test.csv",
        signal,
        detections=detections,
        labels=labels,
        limits=SPCalLimit(1.0, 5.0, 10.0, "Limit", {}),
    )
    assert np.all(result.indicies == [0, 1, 2])
    assert result.events == 50
    assert result.number == 3
    assert result.number_error == np.sqrt(3)
    assert np.isclose(result.background, np.mean(signal[labels == 0]))
    assert np.isclose(result.background_error, np.std(signal[labels == 0]))

    assert result.ionic_background is None
    assert result.number_concentration is None
    assert result.mass_concentration is None

    assert result.asCellConcentration(1.0) is None
    assert result.asMass(1.0) is None
    assert result.asSize(1.0) is None


def test_spcalresult_from_mass_response():
    result = SPCalResult(
        "test.csv",
        signal,
        detections=detections,
        labels=labels,
        limits=SPCalLimit(1.0, 5.0, 10.0, "Limit", {}),
        inputs_kws={
            "density": 0.01,
            "mass_response": 1e-3,
            "mass_fraction": 0.5,
            "cell_diameter": 10e-6,
            "molar_mass": 100.0,
        },
    )
    result.fromMassResponse()

    assert result.ionic_background is None
    assert result.number_concentration is None
    assert result.mass_concentration is None

    assert result.asMass(1.0) == 1.0 * 1e-3 / 0.5
    assert result.asSize(1.0) == particle_size(1.0 * 1e-3 / 0.5, 0.01)

    assert np.all(result.detections["mass"] == result.asMass(detections))
    assert np.all(result.detections["size"] == result.asSize(detections))
    assert np.all(
        result.detections["cell_concentration"]
        == result.asCellConcentration(detections)
    )


def test_spcalresult_from_nebulisation_efficiency():
    result = SPCalResult(
        "test.csv",
        signal,
        detections=detections,
        labels=labels,
        limits=SPCalLimit(1.0, 5.0, 10.0, "Limit", {}),
        inputs_kws={
            "density": 0.01,
            "dwelltime": 1e-3,
            "efficiency": 0.05,
            "uptake": 0.2,
            "response": 100.0,
            "time": 1e-3 * 50,
            "mass_fraction": 0.5,
            "cell_diameter": 10e-6,
            "molar_mass": 100.0,
        },
    )
    result.fromNebulisationEfficiency()

    assert result.ionic_background == result.background / 100.0
    assert result.number_concentration is not None
    assert result.mass_concentration is not None

    assert np.all(result.detections["mass"] == result.asMass(detections))
    assert np.all(result.detections["size"] == result.asSize(detections))
    assert np.all(
        result.detections["cell_concentration"]
        == result.asCellConcentration(detections)
    )


def test_spcalresult_errors():
    with pytest.raises(ValueError):
        SPCalResult(
            "",
            signal,
            detections=np.array([]),
            labels=np.zeros_like(signal),
            limits=SPCalLimit(1.0, 1.0, 1.0, "", {}),
        )

    result = SPCalResult(
        "",
        signal,
        detections=detections,
        labels=labels,
        limits=SPCalLimit(1.0, 1.0, 1.0, "", {}),
    )

    with pytest.raises(KeyError):
        result.convertTo(1.0, "invalid")
    with pytest.raises(ValueError):
        result.fromMassResponse()
    with pytest.raises(ValueError):
        result.fromNebulisationEfficiency()
