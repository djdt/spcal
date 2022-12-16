import tempfile
from pathlib import Path

import numpy as np

from spcal.io import export_single_particle_results, import_single_particle_file
from spcal.limit import SPCalLimit
from spcal.result import SPCalResult

results = {
    "a": SPCalResult(
        "a.csv",
        np.random.random(100),
        np.array([5]),
        np.concatenate((np.zeros(40), np.ones(10), np.zeros(50))),
        SPCalLimit(0.5, 5.0, 10.0, "Limit", {"kw1": 1.0}),
        inputs_kws={"dwelltime": 1e-6, "uptake": 60.0, "notakw": 10.0},
    ),
    "b": SPCalResult(
        "b.csv",
        np.random.random(100),
        np.array([5]),
        np.concatenate((np.zeros(40), np.ones(10), np.zeros(50))),
        SPCalLimit(0.5, 5.0, 10.0, "Limit", {"kw1": 1.0}),
        inputs_kws={
            "dwelltime": 1e-6,
            "uptake": 60.0,
            "notakw": 10.0,
            "density": 10.0,
            "response": 0.1,
            "efficiency": 0.1,
            "mass_fraction": 1.0,
        },
    ),
}
results["b"].fromNebulisationEfficiency()


def test_export_results_arrays():
    tmp = tempfile.mktemp()
    export_single_particle_results(
        tmp, results, output_inputs=False, output_limits=False, output_results=False
    )

    with open(tmp) as fp:
        assert fp.readline().startswith("# SPCal Export")
        assert fp.readline().startswith("# Date")
        assert fp.readline() == "# File,a.csv\n"
        assert fp.readline() == "# Acquisition events,100\n"
        fp.readline()
        fp.readline()
        assert fp.readline() == "a,b,b,b\n"
        assert fp.readline() == "counts,counts,kg,m\n"
        assert fp.readline() == "5,5,0.0003,0.038551464\n"
        fp.readline()
        assert fp.readline() == "# End of export"


test_export_results_arrays()
