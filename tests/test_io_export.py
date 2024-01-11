from pathlib import Path

import numpy as np

from spcal.io.text import export_single_particle_results
from spcal.limit import SPCalLimit
from spcal.result import SPCalResult

results = {
    "a": SPCalResult(
        "a.csv",
        np.ones(100),
        np.array([5, 5, 5, 5, 0]),
        np.concatenate((np.zeros(40), np.ones(10), np.zeros(50))),
        SPCalLimit(0.5, np.array([8.0, 10.0]), "Limit", {"kw1": 1.0, "window": 9}),
        inputs_kws={"dwelltime": 1e-6, "uptake": 1e-3, "not_a_kw": 10.0, "time": 100.0},
    ),
    "b": SPCalResult(
        "b.csv",
        np.full(100, 0.5),
        np.array([0, 9, 9, 9, 9]),
        np.concatenate((np.zeros(40), np.ones(10), np.zeros(50))),
        SPCalLimit(0.5, 9.0, "Limit", {}),
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
clusters = {"signal": np.array([0, 2, 2, 2, 1]), "size": np.array([0, 1, 1, 0, 0])}


def test_export_singleparticle_inputs(tmp_path: Path):
    tmp = tmp_path.joinpath("test_export_inputs.csv")
    export_single_particle_results(
        tmp, results, clusters, output_results=False, output_arrays=False
    )

    with tmp.open("r") as fp:
        assert fp.readline().startswith("# SPCal Export")
        assert fp.readline().startswith("# Date")
        assert fp.readline() == "# File,a.csv\n"
        assert fp.readline() == "# Acquisition events,100\n"
        fp.readline()
        assert fp.readline() == "# Options and inputs,a,b\n"
        assert fp.readline() == "# Cell diameter,,10,μm\n"
        assert fp.readline() == "# Density,,0.01,g/cm3\n"
        assert fp.readline() == "# Dwelltime,0.001,0.001,ms\n"
        assert fp.readline() == "# Efficiency,,0.1,\n"
        assert fp.readline() == "# Mass fraction,,1,\n"
        assert fp.readline() == "# Molar mass,,20000,g/mol\n"
        assert fp.readline() == "# Not a kw,10,10,\n"
        assert fp.readline() == "# Response,,2,counts/(μg/L)\n"
        assert fp.readline() == "# Time,100,100,s\n"
        assert fp.readline() == "# Uptake,60,60,ml/min\n"
        fp.readline()
        assert fp.readline() == "# Limit method,Limit (kw1=1.0;window=9),Limit\n"
        fp.readline()
        assert fp.readline() == "# End of export"

    units = {
        "cell_diameter": ("m", 1.0),
    }
    tmp = tmp_path.joinpath("test_export_inputs_units.csv")
    export_single_particle_results(
        tmp,
        results,
        clusters,
        output_results=False,
        output_arrays=False,
        units_for_inputs=units,
    )

    with tmp.open("r") as fp:
        for i in range(5):
            fp.readline()
        assert fp.readline() == "# Options and inputs,a,b\n"
        assert fp.readline() == "# Cell diameter,,1e-05,m\n"


def test_export_singleparticle_results(tmp_path: Path):
    tmp = tmp_path.joinpath("test_export_results.csv")

    export_single_particle_results(
        tmp, results, clusters, output_inputs=False, output_arrays=False
    )
    with tmp.open("r") as fp:
        for i in range(5):
            fp.readline()
        assert fp.readline() == "# Detection results,a,b\n"
        assert fp.readline() == "# Particle number,4,4\n"
        assert fp.readline() == "# Number error,2,2\n"
        assert fp.readline() == "# Number concentration,,400,#/L\n"
        assert fp.readline() == "# Mass concentration,,1.8e-16,kg/L\n"
        fp.readline()
        assert fp.readline() == "# Background,1,0.5,counts\n"
        assert fp.readline() == "#,,1.6838903e-07,m\n"
        assert fp.readline() == "# Background error,0,0,counts\n"
        assert fp.readline() == "# Ionic background,,2.5e-10,kg/L\n"
        fp.readline()
        assert fp.readline() == "# Mean,a,b\n"
        assert fp.readline() == "#,5,9,counts\n"
        assert fp.readline() == "#,,4.5e-19,kg\n"
        assert fp.readline() == "#,,4.413041e-07,m\n"
        assert fp.readline() == "#,,4.2971835e-08,mol/L\n"

        assert fp.readline() == "# Median,a,b\n"
        assert fp.readline() == "#,5,9,counts\n"
        assert fp.readline() == "#,,4.5e-19,kg\n"
        assert fp.readline() == "#,,4.413041e-07,m\n"
        assert fp.readline() == "#,,4.2971835e-08,mol/L\n"

        assert fp.readline() == "# Limits of detection,a,b\n"
        assert fp.readline() == "#,8 - 10,9,counts\n"
        assert fp.readline() == "#,,4.5e-19,kg\n"
        assert fp.readline() == "#,,4.413041e-07,m\n"
        assert fp.readline() == "#,,4.2971835e-08,mol/L\n"
        fp.readline()
        assert fp.readline() == "# End of export"

    tmp = tmp_path.joinpath("test_export_results_units.csv")
    export_single_particle_results(
        tmp,
        results,
        clusters,
        output_inputs=False,
        output_arrays=False,
        units_for_results={"mass": ("fg", 1e-18)},
    )
    with tmp.open("r") as fp:
        for i in range(9):
            fp.readline()
        assert fp.readline() == "# Mass concentration,,180,fg/L\n"
        for i in range(4):
            fp.readline()
        assert fp.readline() == "# Ionic background,,2.5e+08,fg/L\n"
        for i in range(3):
            fp.readline()
        assert fp.readline() == "#,,0.45,fg\n"


def test_export_singleparticle_arrays(tmp_path: Path):
    tmp = tmp_path.joinpath("test_export_arrays.csv")

    export_single_particle_results(
        tmp, results, clusters, output_inputs=False, output_results=False
    )

    with tmp.open("r") as fp:
        for i in range(5):
            fp.readline()
        fp.readline()
        assert fp.readline() == "a,b,b,b,b\n"
        assert fp.readline() == "counts,counts,kg,m,mol/L\n"
        # Todo, compute these
        assert fp.readline() == "5,,,,\n"
        assert fp.readline() == "5,9,4.5e-19,4.413041e-07,4.2971835e-08\n"
        assert fp.readline() == "5,9,4.5e-19,4.413041e-07,4.2971835e-08\n"
        assert fp.readline() == "5,9,4.5e-19,4.413041e-07,4.2971835e-08\n"
        assert fp.readline() == ",9,4.5e-19,4.413041e-07,4.2971835e-08\n"
        fp.readline()
        assert fp.readline() == "# End of export"

    tmp = tmp_path.joinpath("test_export_arrays_units.csv")
    export_single_particle_results(
        tmp,
        results,
        clusters,
        output_inputs=False,
        output_results=False,
        units_for_results={"signal": ("cts", 1.0), "mass": ("fg", 1e-18)},
    )
    with tmp.open("r") as fp:
        for i in range(7):
            fp.readline()
        assert fp.readline().startswith("cts,cts,fg")
        assert fp.readline().startswith("5,,,,")


def test_export_singleparticle_compositions(tmp_path: Path):
    tmp = tmp_path.joinpath("test_export_arrays.csv")
    export_single_particle_results(
        tmp,
        results,
        clusters,
        output_inputs=False,
        output_results=False,
        output_arrays=False,
        output_compositions=True,
    )

    with tmp.open("r") as fp:
        for i in range(5):
            fp.readline()
        # fp.readline()
        assert fp.readline() == "# Peak composition,count,a,error,b,error\n"
        assert fp.readline() == "# Signal,3,0.3571,0,0.6429,0\n"
        assert fp.readline() == ",1,0,0,1,0\n"
        assert fp.readline() == ",1,1,0,0,0\n"
        assert fp.readline() == "# Mass,1,0,0,1,0\n"
