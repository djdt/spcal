import numpy as np
from pathlib import Path
import pytest

from spcal.processing import options
from spcal.processing.method import SPCalProcessingMethod
from spcal.datafile import SPCalTOFWERKDataFile
from spcal.isotope import ISOTOPE_TABLE, SPCalIsotopeExpression
from spcal.processing.filter import SPCalClusterFilter, SPCalValueFilter


@pytest.fixture(scope="module")
def test_datafile(test_data_path: Path) -> SPCalTOFWERKDataFile:
    return SPCalTOFWERKDataFile.load(
        test_data_path.joinpath("tofwerk/tofwerk_testdata.h5")
    )


def test_spcal_procesing_instrument_options():
    empty = options.SPCalInstrumentOptions(None, None)
    assert empty.canCalibrate("signal")
    assert not empty.canCalibrate("mass")
    assert empty.canCalibrate("mass", mode="mass response")

    full = options.SPCalInstrumentOptions(1.0, 1.0)
    assert full.canCalibrate("signal")
    assert full.canCalibrate("mass")


def test_spcal_processing_isotope_options():
    empty = options.SPCalIsotopeOptions(None, None, None)
    assert empty.canCalibrate("signal")
    assert not empty.canCalibrate("size")
    assert not empty.canCalibrate("mass")

    full = options.SPCalIsotopeOptions(1.0, 1.0, 1.0)
    assert full.canCalibrate("signal")
    assert full.canCalibrate("mass")
    assert full.canCalibrate("size")

    mass = options.SPCalIsotopeOptions(None, None, 1.0, mass_response=1.0)
    assert mass.canCalibrate("signal")
    assert not mass.canCalibrate("size")
    assert not mass.canCalibrate("mass")
    assert mass.canCalibrate("mass", mode="mass response")

    assert empty == empty
    assert empty != full


def test_spcal_processing_limit_options(test_datafile: SPCalTOFWERKDataFile):
    limit_options = options.SPCalLimitOptions()
    limit_options.compound_poisson_kws["sigma"] = 0.65  # lock
    limit_options.manual_limits[ISOTOPE_TABLE[("Ru", 101)]] = 10.0

    for method in ["gaussian", "poisson", "compound poisson", "manual input"]:
        limit = limit_options.limitsForIsotope(
            test_datafile, ISOTOPE_TABLE[("Ru", 101)], limit_method=method
        )
        assert limit.name == method.title()

    limit = limit_options.limitsForIsotope(
        test_datafile, ISOTOPE_TABLE[("Ru", 101)], limit_method="automatic"
    )
    assert limit.name == "Compound Poisson"

    limit = limit_options.limitsForIsotope(
        test_datafile, ISOTOPE_TABLE[("Ru", 101)], limit_method="highest"
    )
    assert limit.name == "Gaussian"

    limit = limit_options.limitsForIsotope(
        test_datafile, ISOTOPE_TABLE[("S", 32)], limit_method="automatic"
    )
    assert limit.name == "Gaussian"

    limit = limit_options.limitsForIsotope(
        test_datafile, ISOTOPE_TABLE[("K", 41)], limit_method="highest"
    )
    assert limit.name == "Compound Poisson"

    # manual
    limit = limit_options.limitsForIsotope(
        test_datafile, ISOTOPE_TABLE[("Ru", 101)], limit_method="manual input"
    )
    assert limit.detection_threshold == 10.0
    limit = limit_options.limitsForIsotope(
        test_datafile, ISOTOPE_TABLE[("K", 41)], limit_method="manual input"
    )
    assert limit.detection_threshold == limit_options.default_manual_limit

    # sia
    limit_options.single_ion_parameters = np.array(
        [(1.0, 2.0, 0.8), (100.0, 2.0, 1.0)],
        dtype=[("mass", float), ("mu", float), ("sigma", float)],
    )
    limit_sia = limit_options.limitsForIsotope(
        test_datafile, ISOTOPE_TABLE[("K", 41)], limit_method="compound poisson"
    )
    assert limit.detection_threshold != limit_sia.detection_threshold

    # sia with expr
    isotope = SPCalIsotopeExpression(
        "test", (ISOTOPE_TABLE[("Ar", 40)], ISOTOPE_TABLE[("K", 41)])
    )
    limit_sia_expr = limit_options.limitsForIsotope(
        test_datafile, isotope, limit_method="compound poisson"
    )
    assert limit_sia_expr.parameters["sigma"] < limit_sia.parameters["sigma"]

    test_datafile.instrument_type = "quadrupole"  # fake for test
    limit = limit_options.limitsForIsotope(
        test_datafile, ISOTOPE_TABLE[("Ru", 101)], limit_method="automatic"
    )
    assert limit.name == "Poisson"

    limit = limit_options.limitsForIsotope(
        test_datafile, ISOTOPE_TABLE[("Ru", 101)], limit_method="highest"
    )
    assert limit.name == "Gaussian"

    # exclusion region
    limit_excluded = limit_options.limitsForIsotope(
        test_datafile,
        ISOTOPE_TABLE[("Ru", 101)],
        limit_method="automatic",
        exclusion_regions=[(10, 20)],
    )
    assert limit.mean_signal != limit_excluded.mean_signal
    test_datafile.instrument_type = "tof"  # fake for test


def test_spcal_processing_method(
    test_datafile: SPCalTOFWERKDataFile, default_method: SPCalProcessingMethod
):
    ru = [ISOTOPE_TABLE[("Ru", x)] for x in [101, 102, 104]]

    method = default_method
    results = method.processDataFile(test_datafile, ru)
    method.filterResults(results)
    assert len(results) == 3

    assert results[ru[0]].number == 8
    assert results[ru[1]].number == 10
    assert results[ru[2]].number == 8
    assert np.isclose(np.mean(results[ru[0]].detections), 274.39)

    assert method.canCalibrate("signal", ru[0])
    assert not method.canCalibrate("mass", ru[0])
    assert not method.canCalibrate("size", ru[0])

    method.instrument_options.uptake = 1.0
    method.instrument_options.efficiency = 1.0
    method.isotope_options[ru[0]] = options.SPCalIsotopeOptions(
        1.0, 1.0, 1.0, mass_response=1.0
    )

    assert method.canCalibrate("mass", ru[0])
    assert method.canCalibrate("size", ru[0])

    assert method.calibrateTo(1.0, "mass", ru[0], 1e-3) == 0.001
    assert method.calibrateTo(1.0, "size", ru[0], 1e-3) == np.cbrt(6.0 / np.pi * 0.001)

    method.calibration_mode = "mass reponse"
    assert method.calibrateTo(1.0, "mass", ru[0], 1e-3) == 1.0

    clusters = method.processClusters(results, "signal")
    assert np.all(clusters == [1, 5, 1, 1, 1, 2, 1, 4, 3, 2])

    # test other accumulation methods
    method.accumulation_method = "half detection threshold"
    results = method.processDataFile(test_datafile, [ru[0]])
    assert np.isclose(np.mean(results[ru[0]].detections), 267.712)

    method.accumulation_method = "detection threshold"
    results = method.processDataFile(test_datafile, [ru[0]])
    assert np.isclose(np.mean(results[ru[0]].detections), 259.619)


def test_spcal_processing_method_exclusions(
    test_datafile: SPCalTOFWERKDataFile, default_method: SPCalProcessingMethod
):
    ru = ISOTOPE_TABLE[("Ru", 101)]

    results = default_method.processDataFile(test_datafile, [ru])
    assert np.isclose(np.mean(results[ru].detections), 274.39)
    default_method.exclusion_regions = [(40.0, 70.0)]
    results = default_method.processDataFile(test_datafile, [ru])
    assert not np.isclose(np.mean(results[ru].detections), 106.943)
    default_method.exclusion_regions = []


def test_spcal_processing_method_filters(
    test_datafile: SPCalTOFWERKDataFile, default_method: SPCalProcessingMethod
):
    ru = [ISOTOPE_TABLE[("Ru", x)] for x in [101, 102, 104]]

    method = default_method
    results = method.processDataFile(test_datafile, ru)

    method.setFilters([[SPCalValueFilter(ru[0], "signal", np.greater, 200.0)]], [[]])

    method.filterResults(results)
    assert results[ru[0]].number == 2

    method.setFilters(
        [[SPCalValueFilter(ru[0], "signal", np.less, 200.0, prefer_invalid=True)]], [[]]
    )

    method.filterResults(results)
    assert results[ru[0]].number == 6

    method.setFilters(
        [
            [
                SPCalValueFilter(ru[0], "signal", np.greater, 70.0),
                SPCalValueFilter(ru[0], "signal", np.less, 100.0, prefer_invalid=True),
            ]
        ],
        [[]],
    )
    method.filterResults(results)
    assert results[ru[0]].number == 2

    # index filter
    method.setFilters([[]], [[SPCalClusterFilter("signal", 1)]])
    method.filterResults(results)

    clusters = {"signal": method.processClusters(results)}
    method.filterIndicies(results, clusters)

    assert results[ru[0]].number == 5

    method.setFilters([[]], [[SPCalClusterFilter("signal", 2)]])
    method.filterIndicies(results, clusters)

    assert results[ru[0]].number == 0

    method.setFilters(
        [[]], [[SPCalClusterFilter("signal", 1)], [SPCalClusterFilter("signal", 3)]]
    )
    method.filterIndicies(results, clusters)

    assert results[ru[0]].number == 6

    # combined
    method.setFilters(
        [[SPCalValueFilter(ru[0], "signal", np.greater, 200.0)]],
        [[SPCalClusterFilter("signal", 1)]],
    )
    method.filterResults(results)
    clusters = {"signal": method.processClusters(results)}
    method.filterIndicies(results, clusters)

    assert results[ru[0]].number == 1

    # not implemented
    # method.setFilters([[SPCalTimeFilter(0.0, 10.0)]], [[]])
    # method.filterResults(results)
    # assert results[ru[0]].number == 4


def test_spcal_processing_results(
    test_datafile: SPCalTOFWERKDataFile, default_method: SPCalProcessingMethod
):
    ru = [ISOTOPE_TABLE[("Ru", x)] for x in [101, 102, 104]]
    method = default_method
    method.isotope_options[ru[0]] = options.SPCalIsotopeOptions(1.0, 1.0, 1.0)
    method.setFilters([[SPCalValueFilter(ru[0], "signal", np.less, 200.0)]], [[]])
    results = method.processDataFile(test_datafile, ru)

    assert results[ru[0]].isotope == ru[0]
    assert results[ru[0]].limit.name == "Compound Poisson"
    assert results[ru[0]].method == method
    assert results[ru[0]].signals.size == results[ru[0]].times.size

    assert np.isclose(results[ru[0]].background, 1.83655)
    assert np.isclose(results[ru[0]].background_error, 1.25618)
    assert results[ru[0]].ionic_background is not None
    assert np.isclose(results[ru[0]].ionic_background, 1.83655)  # type: ignore
    assert results[ru[1]].ionic_background is None

    assert results[ru[0]].num_events == results[ru[0]].signals.size
    assert np.isclose(results[ru[0]].total_time, 90.068)
    assert results[ru[0]].valid_events == results[ru[0]].num_events
    assert results[ru[0]].number == 8

    method.filterResults(results)

    assert np.all(results[ru[0]].filter_indicies == [0, 1, 2, 3, 5, 7])
    assert np.all(results[ru[0]].peak_indicies == [0, 1, 2, 3, 4, 6, 7, 8])
    assert results[ru[0]].number == 6
    assert results[ru[0]].peakValues().size == 10
