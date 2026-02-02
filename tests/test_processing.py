from spcal import processing


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


# def test_spcal_processing_limit_options():
#     gaus = processing.SPCalLimitOptions("gaussian")
