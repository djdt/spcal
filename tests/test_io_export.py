from pathlib import Path
import numpy as np

import pytest
from spcal.datafile import SPCalTextDataFile
from spcal.io import export
from spcal.isotope import ISOTOPE_TABLE
from spcal.processing.method import SPCalProcessingMethod
from spcal.processing.options import SPCalIsotopeOptions


@pytest.fixture(scope="module")
def export_method() -> SPCalProcessingMethod:
    method = SPCalProcessingMethod()
    method.limit_options.compound_poisson_kws["sigma"] = 0.65
    method.instrument_options.uptake = 0.2e-3 / 60.0
    method.instrument_options.efficiency = 0.1
    method.isotope_options[ISOTOPE_TABLE[("Ag", 107)]] = SPCalIsotopeOptions(
        10.49e3, 1e9, 1.0
    )
    method.isotope_options[ISOTOPE_TABLE[("Ag", 109)]] = SPCalIsotopeOptions(
        5.852e3, 2e9, 0.8502
    )
    method.isotope_options[ISOTOPE_TABLE[("Au", 197)]] = SPCalIsotopeOptions(
        19.3e3, 3e9, 1.0
    )
    return method


@pytest.fixture(scope="module")
def export_datafile(test_data_path: Path) -> SPCalTextDataFile:
    df = SPCalTextDataFile.load(
        test_data_path.joinpath("text/tof_mix_au_ag_auag.csv"), skip_rows=1
    )
    df.selected_isotopes = [
        ISOTOPE_TABLE[("Ag", 107)],
        ISOTOPE_TABLE[("Ag", 109)],
        ISOTOPE_TABLE[("Au", 197)],
    ]
    return df


def test_export_spcal_processing_results_known(
    test_data_path: Path,
    tmp_path: Path,
    export_datafile: SPCalTextDataFile,
    export_method: SPCalProcessingMethod,
):
    tmp = tmp_path.joinpath("test_export.csv")

    units = {"signal": ("cts", 1.0), "mass": ("fg", 1e-18), "size": ("nm", 1e-9)}

    results = export_method.processDataFile(export_datafile)
    export_method.filterResults(results)
    clusters = {
        key: export_method.processClusters(results, key)
        for key in ["signal", "mass", "size"]
    }

    export.export_spcal_processing_results(
        tmp,
        export_datafile,
        list(results.values()),
        clusters,
        units,
        export_compositions=True,
    )
    lines = tmp.open("r").readlines()
    lines_checked = (
        test_data_path.joinpath("text_results_checked.csv").open("r").readlines()
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
    test_data_path: Path,
    tmp_path: Path,
    export_datafile: SPCalTextDataFile,
    export_method: SPCalProcessingMethod,
):
    units = {"signal": ("cts", 1.0), "mass": ("fg", 1e-18), "size": ("nm", 1e-9)}

    results = export_method.processDataFile(export_datafile)
    export_method.filterResults(results)

    tmp = tmp_path.joinpath("test_summary.csv")
    with tmp.open("w") as fp:
        export.append_results_summary(
            fp, export_datafile, list(results.values()), units
        )

    lines = tmp.open("r").readlines()
    lines_checked = (
        test_data_path.joinpath("text_batch_checked.csv").open("r").readlines()
    )

    assert lines[0] == "Data File,Isotope,Name,Unit,Value\n"

    for line, checked in zip(lines[1:], lines_checked[1:]):
        tok = line.strip().split(",")[1:]
        tok_checked = checked.strip().split(",")[1:]
        for t, tc in zip(tok[:-1], tok_checked[:-1]):
            assert t == tc
        assert np.isclose(float(tok[-1]), float(tok_checked[-1]), rtol=0.001)
