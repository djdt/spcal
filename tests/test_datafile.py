import pytest
from spcal import datafile
from pathlib import Path
import numpy as np

from spcal.isotope import ISOTOPE_TABLE


def test_spcal_datafile():
    df = datafile.SPCalDataFile(Path(), np.linspace(1, 2, 101), "quadrupole")

    assert df._event_time is None
    assert df.event_time == 0.01
    assert df._event_time is not None  # test cache
    assert df.total_time == 1.01
    assert not df.isTOF()


def test_spcal_datafile_text_agilent(test_data_path: Path):
    path = test_data_path.joinpath("text/agilent_au50nm.csv")
    df = datafile.SPCalTextDataFile.load(path, skip_rows=4)

    assert df.num_events == 9996
    assert np.isclose(df.event_time, 100e-6)
    assert np.isclose(df.total_time, 9996 * 100e-6)
    assert not df.isTOF()
    assert not df.cps

    assert df.isotopes == [ISOTOPE_TABLE[("Au", 197)]]
    assert list(df.isotope_table.values()) == ["Au197"]

    assert np.isclose(np.median(df[df.isotopes[0]]), 1.0)
    assert np.isclose(df[df.isotopes[0]].max(), 439.67)

    df = datafile.SPCalTextDataFile.load(path, skip_rows=4, override_event_time=1.0)
    assert np.isclose(df.event_time, 1.0)
    assert np.isclose(df.total_time, 9996.0)


def test_spcal_datafile_text_agilent_cps(test_data_path: Path):
    path = test_data_path.joinpath("text/agilent_negative_test.csv")
    df = datafile.SPCalTextDataFile.load(path, skip_rows=4, cps=True)

    assert np.all(np.isfinite(df[df.isotopes[0]]))
    assert np.all(df[df.isotopes[0]] >= 0.0)
    assert np.isclose(np.max(df[df.isotopes[0]]), 39.62782)


def test_spcal_datafile_text_icap(test_data_path: Path):
    path = test_data_path.joinpath("text/thermo_icap_export.csv")

    with pytest.raises(NameError):
        datafile.SPCalTextDataFile.load(path, skip_rows=2)

    df = datafile.SPCalTextDataFile.load(
        path,
        skip_rows=2,
        drop_fields=["Number", "Time_80Se_|_80Se.16O"],
        isotope_table={ISOTOPE_TABLE[("Se", 80)]: "Intensity_(cps)_80Se_|_80Se.16O"},
        cps=True,
    )
    assert df.num_events == 1000
    assert np.isclose(df.event_time, 50e-6)
    assert np.isclose(df.total_time, 1000 * 50e-6)
    assert not df.isTOF()
    assert df.cps

    assert df.isotopes == [ISOTOPE_TABLE[("Se", 80)]]

    assert np.isclose(df[df.isotopes[0]].max(), 2.0, atol=0.1)  # check cps


def test_spcal_datafile_text_nu(test_data_path: Path):
    path = test_data_path.joinpath("text/nu_export_auag.csv")

    df = datafile.SPCalTextDataFile.load(
        path,
        skip_rows=1,
        drop_fields=["Time_(ms)"],
        isotope_table={
            ISOTOPE_TABLE[("Ag", 107)]: "106.905_-_seg_Full_mass_spectrum_att_1",
            ISOTOPE_TABLE[("Ag", 109)]: "108.905_-_seg_Full_mass_spectrum_att_1",
            ISOTOPE_TABLE[("Au", 197)]: "196.967_-_seg_Full_mass_spectrum_att_1",
        },
    )
    assert df.times.size == 999
    assert df.num_events == 999
    assert np.isclose(df.event_time, 4.852e-5)
    assert np.isclose(df.total_time, 999 * 4.852e-5)
    assert df.isTOF()

    assert np.isclose(np.mean(df[ISOTOPE_TABLE[("Ag", 107)]]), 0.006001689)
    assert np.isclose(np.mean(df[ISOTOPE_TABLE[("Ag", 109)]]), 0.004975563)
    assert np.isclose(np.mean(df[ISOTOPE_TABLE[("Au", 197)]]), 0.002678757)


def test_spcal_datafile_text_tofwerk(test_data_path: Path):
    path = test_data_path.joinpath("text/tofwerk_export_au.csv")
    df = datafile.SPCalTextDataFile.load(
        path,
        skip_rows=1,
        drop_fields=["Index", "timestamp_(s)"],
        isotope_table={ISOTOPE_TABLE[("Au", 197)]: "[197Au]+_(cts)"},
    )

    assert df.num_events == 999
    assert np.isclose(df.event_time, 0.9999e-3)
    assert np.isclose(df.total_time, 999 * 0.9999e-3)

    assert np.isclose(np.mean(df[ISOTOPE_TABLE[("Au", 197)]]), 2.142439)


def test_spcal_datafile_nu(test_data_path: Path):
    path = test_data_path.joinpath("nu/run.info")
    df = datafile.SPCalNuDataFile.load(path)

    assert df.times.size == 40
    assert df.num_events == 40
    assert df.event_time == 9.824e-5

    assert len(df.masses) == 127
    assert df.signals.shape == (40, 127)
    assert len(df.isotopes) == 188

    assert np.isclose(np.nanmean(df[ISOTOPE_TABLE[("Au", 197)]]), 3.9454572)
    assert df.isotope_table[ISOTOPE_TABLE[("Au", 197)]] == 114


def test_spcal_datafile_nu_integ_range(test_data_path: Path):
    path = test_data_path.joinpath("nu")
    df = datafile.SPCalNuDataFile.load(path, first_integ_file=1, last_integ_file=2)
    assert not np.any(np.isnan(df.signals))
    assert df.times.size == 10
    assert df.num_events == 10


def test_spcal_datafile_tofwerk(test_data_path: Path):
    path = test_data_path.joinpath("tofwerk/tofwerk_testdata.h5")
    df = datafile.SPCalTOFWERKDataFile.load(path)

    assert df.signals.shape[-1] == 315
    assert df.masses.shape[0] == 315
    assert len(df.isotopes) == 278  # ions are removed

    assert df.num_events == 89 * 11 * 5
    # assert np.isclose(df.event_time, 9.2e-5)  # wrong because I have messed with the data

    assert np.isclose(np.mean(df[ISOTOPE_TABLE[("Ru", 101)]]), 2.2688558)
