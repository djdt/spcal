import numpy as np
from pathlib import Path

from spcal import processing
from spcal.datafile import SPCalTOFWERKDataFile
from spcal.isotope import ISOTOPE_TABLE, SPCalIsotopeExpression
from spcal.processing.filter import (
    SPCalProcessingFilter,
    SPCalTimeFilter,
    SPCalValueFilter,
)


def test_spcal_procesing_instrument_options():
    empty = processing.SPCalInstrumentOptions(None, None)
    assert empty.canCalibrate("signal")
    assert not empty.canCalibrate("mass")
    assert empty.canCalibrate("mass", mode="mass response")

    full = processing.SPCalInstrumentOptions(1.0, 1.0)
    assert full.canCalibrate("signal")
    assert full.canCalibrate("mass")


def test_spcal_processing_isotope_options():
    empty = processing.SPCalIsotopeOptions(None, None, None)
    assert empty.canCalibrate("signal")
    assert not empty.canCalibrate("size")
    assert not empty.canCalibrate("mass")

    full = processing.SPCalIsotopeOptions(1.0, 1.0, 1.0)
    assert full.canCalibrate("signal")
    assert full.canCalibrate("mass")
    assert full.canCalibrate("size")

    mass = processing.SPCalIsotopeOptions(None, None, 1.0, mass_response=1.0)
    assert mass.canCalibrate("signal")
    assert not mass.canCalibrate("size")
    assert not mass.canCalibrate("mass")
    assert mass.canCalibrate("mass", mode="mass response")

    assert empty == empty
    assert empty != full


def test_spcal_processing_limit_options(test_data_path: Path):
    df = SPCalTOFWERKDataFile.load(
        test_data_path.joinpath("tofwerk/tofwerk_testdata.h5")
    )

    options = processing.SPCalLimitOptions()

    for method in ["gaussian", "poisson", "compound poisson"]:
        limit = options.limitsForIsotope(
            df, ISOTOPE_TABLE[("Au", 197)], limit_method=method
        )
        assert limit.name == method.title()

    limit = options.limitsForIsotope(
        df, ISOTOPE_TABLE[("Au", 197)], limit_method="automatic"
    )
    assert limit.name == "Compound Poisson"

    limit = options.limitsForIsotope(
        df, ISOTOPE_TABLE[("Au", 197)], limit_method="highest"
    )
    assert limit.name == "Compound Poisson"

    limit = options.limitsForIsotope(
        df, ISOTOPE_TABLE[("K", 41)], limit_method="automatic"
    )
    assert limit.name == "Gaussian"

    limit = options.limitsForIsotope(
        df, ISOTOPE_TABLE[("K", 41)], limit_method="highest"
    )
    assert limit.name == "Compound Poisson"

    # sia
    options.single_ion_parameters = np.array(
        [(1.0, 2.0, 0.8), (100.0, 2.0, 1.0)],
        dtype=[("mass", float), ("mu", float), ("sigma", float)],
    )
    limit_sia = options.limitsForIsotope(
        df, ISOTOPE_TABLE[("K", 41)], limit_method="compound poisson"
    )
    assert limit.detection_threshold != limit_sia.detection_threshold

    # sia with expr
    isotope = SPCalIsotopeExpression(
        "test", (ISOTOPE_TABLE[("Ar", 40)], ISOTOPE_TABLE[("K", 41)])
    )
    limit_sia_expr = options.limitsForIsotope(
        df, isotope, limit_method="compound poisson"
    )
    assert limit_sia_expr.parameters["sigma"] < limit_sia.parameters["sigma"]

    df.instrument_type = "quadrupole"  # fake for test
    limit = options.limitsForIsotope(
        df, ISOTOPE_TABLE[("Au", 197)], limit_method="automatic"
    )
    assert limit.name == "Poisson"

    limit = options.limitsForIsotope(
        df, ISOTOPE_TABLE[("Au", 197)], limit_method="highest"
    )
    assert limit.name == "Poisson"

    # exclusion region
    limit_excluded = options.limitsForIsotope(
        df,
        ISOTOPE_TABLE[("Au", 197)],
        limit_method="automatic",
        exclusion_regions=[(10, 20)],
    )
    assert limit.mean_signal != limit_excluded.mean_signal
    df.instrument_type = "tof"  # fake for test


def test_spcal_processing_method(test_data_path: Path):
    df = SPCalTOFWERKDataFile.load(
        test_data_path.joinpath("tofwerk/tofwerk_testdata.h5")
    )
    indium = ISOTOPE_TABLE[("In", 115)]
    tin = ISOTOPE_TABLE[("Sn", 118)]

    method = processing.SPCalProcessingMethod()
    results = method.processDataFile(df, [indium, tin])
    results = method.filterResults(results)
    assert len(results) == 2

    assert results[indium].number == 6
    assert results[tin].number == 2
    assert np.isclose(np.mean(results[indium].detections), 138.305)

    assert method.canCalibrate("signal", indium)
    assert not method.canCalibrate("mass", indium)
    assert not method.canCalibrate("size", indium)

    method.instrument_options.uptake = 1.0
    method.instrument_options.efficiency = 1.0
    method.isotope_options[indium] = processing.SPCalIsotopeOptions(
        1.0, 1.0, 1.0, mass_response=1.0
    )

    assert method.canCalibrate("mass", indium)
    assert method.canCalibrate("size", indium)

    assert method.calibrateTo(1.0, "mass", indium, 1e-3) == 0.001
    assert method.calibrateTo(1.0, "size", indium, 1e-3) == np.cbrt(6.0 / np.pi * 0.001)

    method.calibration_mode = "mass reponse"
    assert method.calibrateTo(1.0, "mass", indium, 1e-3) == 1.0

    clusters = method.processClusters(results, "signal")
    assert np.all(clusters == [1, 1, 1, 2, 1, 3])

    # test other accumulation methods
    method.accumulation_method = "half detection threshold"
    results = method.processDataFile(df, [indium])
    assert np.isclose(np.mean(next(iter(results.values())).detections), 135.354)

    method.accumulation_method = "detection threshold"
    results = method.processDataFile(df, [indium])
    assert np.isclose(np.mean(next(iter(results.values())).detections), 133.014)


def test_spcal_processing_method_filters(test_data_path: Path):
    df = SPCalTOFWERKDataFile.load(
        test_data_path.joinpath("tofwerk/tofwerk_testdata.h5")
    )
    indium = ISOTOPE_TABLE[("In", 115)]
    tin = ISOTOPE_TABLE[("Sn", 118)]

    method = processing.SPCalProcessingMethod()
    results = method.processDataFile(df, [indium, tin])

    method.setFilters([[SPCalValueFilter(indium, "signal", np.greater, 200.0)]])

    results = method.filterResults(results)
    assert results[indium].number == 1

    method.setFilters([[SPCalValueFilter(indium, "signal", np.less, 200.0)]])

    results = method.filterResults(results)
    assert results[indium].number == 5

    method.setFilters(
        [
            [
                SPCalValueFilter(indium, "signal", np.greater, 10.0),
                SPCalValueFilter(indium, "signal", np.less, 25.0),
            ]
        ]
    )
    results = method.filterResults(results)
    assert results[indium].number == 1

    method.setFilters(
        [
            [SPCalValueFilter(indium, "signal", np.greater, 10.0)],
            [SPCalValueFilter(indium, "signal", np.less, 25.0)],
        ]
    )
    results = method.filterResults(results)
    assert results[indium].number == 6

    method.setFilters([[SPCalTimeFilter(0.0, 10.0)]])
    results = method.filterResults(results)
    assert results[indium].number == 4


def test_spcal_processing_results(test_data_path: Path):
    df = SPCalTOFWERKDataFile.load(
        test_data_path.joinpath("tofwerk/tofwerk_testdata.h5")
    )
    indium = ISOTOPE_TABLE[("In", 115)]
    tin = ISOTOPE_TABLE[("Sn", 118)]

    method = processing.SPCalProcessingMethod()
    method.isotope_options[indium] = processing.SPCalIsotopeOptions(1.0, 1.0, 1.0)
    method.setFilters([[SPCalValueFilter(indium, "signal", np.less, 200.0)]])
    results = method.processDataFile(df, [indium, tin])

    assert results[indium].isotope == indium
    assert results[indium].limit.name == "Compound Poisson"
    assert results[indium].method == method
    assert results[indium].signals.size == results[indium].times.size

    assert np.isclose(results[indium].background, 0.577764)
    assert np.isclose(results[indium].background_error, 0.62087)
    assert results[indium].ionic_background is not None
    assert np.isclose(results[indium].ionic_background, 0.577764)  # type: ignore
    assert results[tin].ionic_background is None

    assert results[indium].num_events == results[indium].signals.size
    assert np.isclose(results[indium].total_time, 90.068)
    assert results[indium].valid_events == results[indium].num_events
    assert results[indium].number == 6

    results = method.filterResults(results)

    assert np.all(results[indium].filter_indicies == [0, 1, 2, 3, 4])
    assert np.all(results[indium].peak_indicies == [0, 1, 2, 3, 4, 5])
    assert results[indium].number == 5
    assert results[indium].peakValues().size == 6
