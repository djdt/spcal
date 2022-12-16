import tempfile
from pathlib import Path

import numpy as np

from spcal.io import export_single_particle_results, import_single_particle_file
from spcal.limit import SPCalLimit
from spcal.result import SPCalResult

results = {
    "a": SPCalResult(
        "a.csv",
        np.ones(100),
        np.array([5]),
        np.concatenate((np.zeros(40), np.ones(10), np.zeros(50))),
        SPCalLimit(
            0.5, 5.0, np.array([8.0, 10.0]), "Limit", {"kw1": 1.0}, window_size=9
        ),
        inputs_kws={"dwelltime": 1e-6, "uptake": 1e-3, "not_a_kw": 10.0, "time": 100.0},
    ),
    "b": SPCalResult(
        "b.csv",
        np.full(100, 0.5),
        np.array([9]),
        np.concatenate((np.zeros(40), np.ones(10), np.zeros(50))),
        SPCalLimit(0.5, 5.0, 9.0, "Limit", {}),
        inputs_kws={
            "cell_diameter": 10e-6,
            "dwelltime": 1e-6,
            "uptake": 1e-3,
            "not_a_kw": 10.0,
            "density": 10.0,
            "response": 2e9,
            "efficiency": 0.1,
            "mass_fraction": 1.0,
            "molar_mass": 20.0,
            "time": 100.0,
        },
    ),
}
results["b"].fromNebulisationEfficiency()


def test_export_singleparticle_inputs():
    with tempfile.NamedTemporaryFile(mode="w+") as tmp:
        export_single_particle_results(
            tmp.name, results, output_results=False, output_arrays=False
        )

        assert tmp.readline().startswith("# SPCal Export")
        assert tmp.readline().startswith("# Date")
        assert tmp.readline() == "# File,a.csv\n"
        assert tmp.readline() == "# Acquisition events,100\n"
        tmp.readline()
        assert tmp.readline() == "# Options and inputs,a,b\n"
        assert tmp.readline() == "# Cell diameter,,10,μm\n"
        assert tmp.readline() == "# Density,,0.01,g/cm3\n"
        assert tmp.readline() == "# Dwelltime,0.001,0.001,ms\n"
        assert tmp.readline() == "# Efficiency,,0.1,\n"
        assert tmp.readline() == "# Mass fraction,,1,\n"
        assert tmp.readline() == "# Molar mass,,20000,g/mol\n"
        assert tmp.readline() == "# Not a kw,10,10,\n"
        assert tmp.readline() == "# Response,,2,counts/(μg/L)\n"
        assert tmp.readline() == "# Time,100,100,s\n"
        assert tmp.readline() == "# Uptake,60,60,ml/min\n"
        tmp.readline()
        assert tmp.readline() == "# Limit method,Limit (kw1=1;window=9),Limit\n"
        tmp.readline()
        assert tmp.readline() == "# End of export"

    units = {
        "cell_diameter": ("m", 1.0),
    }
    with tempfile.NamedTemporaryFile(mode="w+") as tmp:
        export_single_particle_results(
            tmp.name,
            results,
            output_results=False,
            output_arrays=False,
            units_for_inputs=units,
        )

        for i in range(5):
            tmp.readline()
        assert tmp.readline() == "# Options and inputs,a,b\n"
        assert tmp.readline() == "# Cell diameter,,1e-05,m\n"


def test_export_singleparticle_results():
    with tempfile.NamedTemporaryFile(mode="w+") as tmp:
        export_single_particle_results(
            tmp.name, results, output_inputs=False, output_arrays=False
        )
        for i in range(5):
            tmp.readline()
        assert tmp.readline() == "# Detection results,a,b\n"
        assert tmp.readline() == "# Particle number,1,1\n"
        assert tmp.readline() == "# Number error,1,1\n"
        assert tmp.readline() == "# Number concentration,,100,#/L\n"
        assert tmp.readline() == "# Mass concentration,,4.5e-17,kg/L\n"
        tmp.readline()
        assert tmp.readline() == "# Background,1,0.5,counts\n"
        assert tmp.readline() == "#,,1.6838903e-07,m\n"
        assert tmp.readline() == "# Background error,0,0,counts\n"
        assert tmp.readline() == "# Ionic background,,2.5e-10,kg/L\n"
        tmp.readline()
        assert tmp.readline() == "# Mean,a,b\n"
        assert tmp.readline() == "#,5,9,counts\n"
        assert tmp.readline() == "#,,4.5e-19,kg\n"
        assert tmp.readline() == "#,,4.413041e-07,m\n"
        assert tmp.readline() == "#,,4.2971835e-08,mol/L\n"

        assert tmp.readline() == "# Median,a,b\n"
        assert tmp.readline() == "#,5,9,counts\n"
        assert tmp.readline() == "#,,4.5e-19,kg\n"
        assert tmp.readline() == "#,,4.413041e-07,m\n"
        assert tmp.readline() == "#,,4.2971835e-08,mol/L\n"

        assert tmp.readline() == "# Limits of detection,a,b\n"
        assert tmp.readline() == "#,8 - 10,9,counts\n"
        assert tmp.readline() == "#,,4.5e-19,kg\n"
        assert tmp.readline() == "#,,4.413041e-07,m\n"
        assert tmp.readline() == "#,,4.2971835e-08,mol/L\n"
        tmp.readline()
        assert tmp.readline() == "# End of export"

    with tempfile.NamedTemporaryFile(mode="w+") as tmp:
        export_single_particle_results(
            tmp.name,
            results,
            output_inputs=False,
            output_arrays=False,
            units_for_results={"mass": ("fg", 1e-18)},
        )
        for i in range(9):
            tmp.readline()
        assert tmp.readline() == "# Mass concentration,,45,fg/L\n"
        for i in range(4):
            tmp.readline()
        assert tmp.readline() == "# Ionic background,,2.5e+08,fg/L\n"
        for i in range(3):
            tmp.readline()
        assert tmp.readline() == "#,,0.45,fg\n"


def test_export_singleparticle_arrays():
    with tempfile.NamedTemporaryFile(mode="w+") as tmp:
        export_single_particle_results(
            tmp.name, results, output_inputs=False, output_results=False
        )

        for i in range(5):
            tmp.readline()
        tmp.readline()
        assert tmp.readline() == "a,b,b,b,b\n"
        assert tmp.readline() == "counts,counts,kg,m,mol/L\n"
        # Todo, compute these
        assert tmp.readline() == "5,9,4.5e-19,4.413041e-07,4.2971835e-08\n"
        tmp.readline()
        assert tmp.readline() == "# End of export"

    with tempfile.NamedTemporaryFile(mode="w+") as tmp:
        export_single_particle_results(
            tmp.name,
            results,
            output_inputs=False,
            output_limits=False,
            output_results=False,
            units_for_results={"signal": ("cts", 1.0), "mass": ("fg", 1e-18)},
        )
        for i in range(7):
            tmp.readline()
        assert tmp.readline().startswith("cts,cts,fg")
        assert tmp.readline().startswith("5,9,0.45")
