from datetime import datetime
import numpy as np
from typing import Any
from pathlib import Path

import json
from importlib.metadata import version

from spcal.datafile import (
    SPCalDataFile,
    SPCalNuDataFile,
    SPCalTOFWERKDataFile,
    SPCalTextDataFile,
)
from spcal.isotope import SPCalIsotope, SPCalIsotopeExpression
from spcal.processing.filter import (
    SPCalClusterFilter,
    SPCalValueFilter,
)
from spcal.processing.method import SPCalProcessingMethod
from spcal.processing.options import (
    SPCalInstrumentOptions,
    SPCalIsotopeOptions,
    SPCalLimitOptions,
)
from spcal.io.session import save_session_json, load_session_json


def test_session_save_load(test_data_path: Path, tmp_path: Path):
    params = np.empty(
        100, dtype=[("mass", np.float32), ("mu", np.float32), ("sigma", np.float32)]
    )
    params["mass"] = np.arange(100)
    params["mu"] = np.random.random(100)
    params["sigma"] = np.random.random(100)

    method = SPCalProcessingMethod()
    method.instrument_options.uptake = 1.0
    method.instrument_options.efficiency = 0.1
    method.expressions = [
        SPCalIsotopeExpression(
            "test_expr",
            ("+", SPCalIsotope.fromString("107Ag"), SPCalIsotope.fromString("109Ag")),
        )
    ]
    method.isotope_options[SPCalIsotope.fromString("197Au")] = SPCalIsotopeOptions(
        1.0, 2.0, 3.0
    )
    method.isotope_options[method.expressions[0]] = SPCalIsotopeOptions(1.0, 2.0, 3.0)
    method.limit_options.manual_limits = {SPCalIsotope.fromString("107Ag"): 10.2}
    method.limit_options.single_ion_parameters = params
    method.result_filters = [
        [
            SPCalValueFilter(
                SPCalIsotope.fromString("107Ag"), "signal", np.greater, 2.0
            ),
            SPCalValueFilter(
                SPCalIsotope.fromString("107Ag"), "signal", np.greater, 3.0
            ),
        ],
        [SPCalValueFilter(SPCalIsotope.fromString("107Ag"), "signal", np.greater, 4.0)],
    ]
    method.index_filters = [[SPCalClusterFilter("signal", 0)]]
    method.exclusion_regions = [(0.4, 1.0), (230.2, 276.0)]

    files = [
        SPCalNuDataFile.load(Path("/home/tom/Downloads/NT032/14-37-30 1 ppb att")),
        SPCalTextDataFile.load(
            Path("/home/tom/Downloads/019SMPL-48-64Ti_count.csv"),
            skip_rows=5,
            isotope_table={SPCalIsotope.fromString("48Ti"): "Ti48_->_64"},
        ),
    ]
    files[0].selected_isotopes = [SPCalIsotope.fromString("197Au")]

    save_session_json(tmp_path.joinpath("session.spcal.json"), method, files)
    method2, files2 = load_session_json(tmp_path.joinpath("session.spcal.json"))

    assert method.instrument_options == method2.instrument_options
    assert method.expressions == method2.expressions
    assert method.result_filters == method2.result_filters
    assert method.index_filters == method2.index_filters
    assert method.isotope_options == method2.isotope_options
    assert method.exclusion_regions == method2.exclusion_regions
