import numpy as np
import pytest

from spcal.limit import SPCalLimit
from spcal.particle import particle_size
from spcal.result import ClusterFilter, Filter, SPCalResult

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
        limits=SPCalLimit(1.0, 5.0, "Limit", {}),
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

    assert not result.canCalibrate("cell_concentration")
    assert not result.canCalibrate("mass")
    assert not result.canCalibrate("size")
    assert not result.canCalibrate("volume")


def test_spcalresult_no_detections():
    result = SPCalResult(
        "test.csv",
        signal,
        detections=np.array([]),
        labels=labels,
        limits=SPCalLimit(1.0, 5.0, "Limit", {}),
    )
    assert result.number == 0
    assert result.number_error == 0


def test_spcalresult_from_mass_response():
    result = SPCalResult(
        "test.csv",
        signal,
        detections=detections,
        labels=labels,
        limits=SPCalLimit(1.0, 5.0, "Limit", {}),
        inputs_kws={
            "density": 0.01,
            "mass_response": 1e-3,
            "mass_fraction": 0.5,
            "cell_diameter": 10e-6,
            "molar_mass": 100.0,
        },
        calibration_mode="mass response",
    )

    assert result.ionic_background is None
    assert result.number_concentration is None
    assert result.mass_concentration is None

    assert result.asMass(1.0) == 1.0 * 1e-3 / 0.5
    assert result.asSize(1.0) == particle_size(1.0 * 1e-3 / 0.5, 0.01)

    assert np.all(
        result.calibrated("cell_concentration", use_indicies=False)
        == result.asCellConcentration(detections)
    )
    assert np.all(
        result.calibrated("mass", use_indicies=False) == result.asMass(detections)
    )
    assert np.all(
        result.calibrated("size", use_indicies=False) == result.asSize(detections)
    )
    assert np.all(
        result.calibrated("volume", use_indicies=False) == result.asVolume(detections)
    )


def test_spcalresult_from_nebulisation_efficiency():
    result = SPCalResult(
        "test.csv",
        signal,
        detections=detections,
        labels=labels,
        limits=SPCalLimit(1.0, 5.0, "Limit", {}),
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
        calibration_mode="efficiency",
    )

    assert result.ionic_background == result.background / 100.0
    assert result.number_concentration is not None
    assert result.mass_concentration is not None

    assert np.all(
        result.calibrated("cell_concentration", use_indicies=False)
        == result.asCellConcentration(detections)
    )
    assert np.all(
        result.calibrated("mass", use_indicies=False) == result.asMass(detections)
    )
    assert np.all(
        result.calibrated("size", use_indicies=False) == result.asSize(detections)
    )
    assert np.all(
        result.calibrated("volume", use_indicies=False) == result.asVolume(detections)
    )


def test_spcalresult_errors():
    # with pytest.raises(ValueError):
    #     SPCalResult(
    #         "",
    #         signal,
    #         detections=np.array([]),
    #         labels=np.zeros_like(signal),
    #         limits=SPCalLimit(1.0, 1.0, "", {}),
    #     )
    with pytest.raises(ValueError):
        SPCalResult(
            "",
            signal,
            detections=detections,
            labels=labels,
            limits=SPCalLimit(1.0, 1.0, "", {}),
            calibration_mode="invalid",
        )

    result = SPCalResult(
        "",
        signal,
        detections=detections,
        labels=labels,
        limits=SPCalLimit(1.0, 1.0, "", {}),
    )

    with pytest.raises(KeyError):
        result.convertTo(1.0, "invalid")


def test_spcalresult_filters_and_valid():
    results = {
        "a": SPCalResult(
            "a.csv",
            np.ones(10),
            np.array([1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0]),
            np.array([1, 0, 2, 0, 3, 0, 4, 0, 5, 5]),
            limits=SPCalLimit(0.1, 0.5, "Limit", {}),
        ),
        "b": SPCalResult(
            "b.csv",
            np.ones(10),
            np.array([0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 5.0]),
            np.array([0, 1, 2, 0, 0, 3, 4, 0, 5, 5]),
            limits=SPCalLimit(0.1, 0.5, "Limit", {}),
            inputs_kws={
                "dwelltime": 1.0,
                "efficiency": 1.0,
                "uptake": 1.0,
                "response": 1.0,
                "mass_fraction": 1.0,
            },
        ),
    }

    assert SPCalResult.all_valid_indicies([]).size == 0
    assert np.all(results["a"].indicies == [0, 2, 4, 6, 8])
    assert np.all(results["b"].indicies == [1, 2, 5, 6, 9])
    assert np.all(
        SPCalResult.all_valid_indicies(list(results.values()))
        == [0, 1, 2, 4, 5, 6, 8, 9]
    )

    filters = [[Filter("a", "signal", ">", 4.0)]]
    idx = Filter.filter_results(filters, results)
    assert idx == [8]

    filters = [[Filter("a", "signal", "<", 4.0), Filter("b", "signal", ">", 2.0)]]
    idx = Filter.filter_results(filters, results)
    assert np.all(idx == [5, 9])

    filters = [[Filter("a", "signal", "<", 4.0), Filter("b", "mass", ">", 2.0)]]
    idx = Filter.filter_results(filters, results)
    assert np.all(idx == [5, 9])

    filters = [[Filter("a", "signal", "==", 1.0)], [Filter("b", "signal", "==", 1.0)]]
    idx = Filter.filter_results(filters, results)
    assert np.all(idx == [0, 1])

    idx = Filter.filter_results([], results)
    assert np.all(idx == np.arange(10))
