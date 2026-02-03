import numpy as np
from pathlib import Path

from spcal import processing
from spcal.datafile import SPCalTOFWERKDataFile
from spcal.isotope import ISOTOPE_TABLE, SPCalIsotopeExpression


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
