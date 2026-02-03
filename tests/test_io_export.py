from pathlib import Path
import numpy as np

import pytest
from spcal.datafile import SPCalTOFWERKDataFile
from spcal.io import export
from spcal.isotope import ISOTOPE_TABLE
from spcal.processing import SPCalProcessingMethod
from spcal.processing.options import SPCalIsotopeOptions


@pytest.fixture(scope="module")
def export_method() -> SPCalProcessingMethod:
    method = SPCalProcessingMethod()
    method.limit_options.compound_poisson_kws["sigma"] = 0.4
    method.instrument_options.uptake = 0.2e-3 / 60.0
    method.instrument_options.efficiency = 0.1
    method.isotope_options[ISOTOPE_TABLE[("In", 115)]] = SPCalIsotopeOptions(
        5.775e3, 1e6, 0.4853
    )
    method.isotope_options[ISOTOPE_TABLE[("Sn", 118)]] = SPCalIsotopeOptions(
        6.85e3, 2e6, 0.2913
    )
    method.isotope_options[ISOTOPE_TABLE[("Sb", 121)]] = SPCalIsotopeOptions(
        5.775e3, 3e6, 0.5146
    )
    return method


def test_export_spcal_processing_results_known(
    test_data_path: Path, tmp_path: Path, export_method: SPCalProcessingMethod
):
    tmp = tmp_path.joinpath("test_export.csv")

    units = {"signal": ("cts", 1.0), "mass": ("fg", 1e-18), "size": ("nm", 1e-9)}

    df = SPCalTOFWERKDataFile.load(
        test_data_path.joinpath("tofwerk/tofwerk_testdata.h5")
    )
    df.selected_isotopes = [
        ISOTOPE_TABLE[("In", 115)],
        ISOTOPE_TABLE[("Sn", 118)],
        ISOTOPE_TABLE[("Sb", 121)],
    ]

    results = export_method.processDataFile(df)
    results = export_method.filterResults(results)
    clusters = {
        key: export_method.processClusters(results, key)
        for key in ["signal", "mass", "size"]
    }

    export.export_spcal_processing_results(
        tmp, df, list(results.values()), clusters, units, export_compositions=True
    )
    lines = tmp.open("r").readlines()
    lines_checked = (
        test_data_path.joinpath("tofwerk_testdata_results_checked.csv")
        .open("r")
        .readlines()
    )

    assert lines[0].startswith("# SPCal Export")
    assert lines[1].startswith("# Date")
    assert lines[2].startswith("# File")

    for line, checked in zip(lines[3:], lines_checked[3:]):
        for tok, tok_checked in zip(line.split(","), checked.split(",")):
            try:
                assert np.isclose(float(tok), float(tok_checked))
            except ValueError:
                assert tok == tok_checked


def test_export_spcal_append_summary(
    test_data_path: Path, tmp_path: Path, export_method: SPCalProcessingMethod
):
    units = {"signal": ("cts", 1.0), "mass": ("fg", 1e-18), "size": ("nm", 1e-9)}
    df = SPCalTOFWERKDataFile.load(
        test_data_path.joinpath("tofwerk/tofwerk_testdata.h5")
    )
    df.selected_isotopes = [
        ISOTOPE_TABLE[("In", 115)],
        ISOTOPE_TABLE[("Sn", 118)],
        ISOTOPE_TABLE[("Sb", 121)],
    ]

    results = export_method.processDataFile(df)
    results = export_method.filterResults(results)

    tmp = tmp_path.joinpath("test_summary.csv")
    with tmp.open("w") as fp:
        export.append_results_summary(fp, df, list(results.values()), units)

    lines = tmp.open("r").readlines()
    lines_checked = (
        test_data_path.joinpath("tofwerk_batch_summary_checked.csv")
        .open("r")
        .readlines()
    )

    assert lines[0] == "Data File,Isotope,Name,Unit,Value\n"

    for line, checked in zip(lines[1:], lines_checked[1:]):
        tok = line.strip().split(",")[1:]
        tok_checked = checked.strip().split(",")[1:]
        for t, tc in zip(tok[:-1], tok_checked[:-1]):
            assert t == tc
        assert np.isclose(float(tok[-1]), float(tok_checked[-1]), rtol=0.001)
