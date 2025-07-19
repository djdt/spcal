from pathlib import Path

import numpy as np
import pytest

from spcal.io.text import (
    export_single_particle_results,
    guess_text_parameters,
    is_text_file,
    read_single_particle_file,
)
from spcal.limit import SPCalLimit
from spcal.result import SPCalResult

results = {
    "a": SPCalResult(
        "a.csv",
        np.ones(100),
        np.array([5.0, 5, 5, 5, 0]),
        np.concatenate((np.zeros(40), np.ones(10), np.zeros(50))),
        SPCalLimit(0.5, np.array([8.0, 10.0]), "Limit", {"kw1": 1.0, "window": 9}),
        inputs_kws={
            "dwelltime": 1e-6,
            "uptake": 1e-3,
            "not_a_kw": 10.0,
            "time": 100.0,
        },
    ),
    "b": SPCalResult(
        "b.csv",
        np.full(100, 0.5),
        np.array([0.0, 8, 9, 9, 10]),
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
clusters = {"signal": np.array([0, 2, 2, 2, 1]), "mass": np.array([0, 1, 1, 0, 0])}
times = np.array([0.1, 0.2, 0.3, 0.4, 0.5])


def test_io_is_text_file():
    assert is_text_file(Path(__file__).parent.joinpath("data/text/text_normal.csv"))
    assert not is_text_file(Path(__file__).parent.joinpath("data/text/text_normal.bad"))
    assert not is_text_file(Path(__file__).parent.joinpath("data/text/fake.txt"))


def test_io_text_import():
    path = Path(__file__).parent.joinpath("data/text/text_normal.csv")
    with pytest.warns():
        data = read_single_particle_file(
            path, first_line=2, columns=(0, 1, 2), convert_cps=10.0
        )
    assert np.all(data["A"] == 10)
    assert np.all(data["B"] == [10, 20, 30])
    assert np.all(data["C"] == [10, 20, 40])


def test_io_text_import_one_column():
    path = Path(__file__).parent.joinpath("data/text/text_normal.csv")
    data = read_single_particle_file(path, first_line=2, columns=(0,))
    # bad column is imported
    assert np.all(data["A"] == [1, 1, 0, 1])

    path = Path(__file__).parent.joinpath("data/text/text_onecol.csv")
    data = read_single_particle_file(path)
    assert np.all(data["A"] == [1, 2, 3, 4, 5])


def test_io_text_import_no_columns():
    path = Path(__file__).parent.joinpath("data/text/text_normal.csv")
    with pytest.warns():
        data = read_single_particle_file(path, first_line=2, convert_cps=10.0)
    assert np.all(data["A"] == 10)
    assert np.all(data["B"] == [10, 20, 30])
    assert np.all(data["C"] == [10, 20, 40])


def test_io_text_import_euro():
    path = Path(__file__).parent.joinpath("data/text/text_euro.csv")
    data = read_single_particle_file(
        path, delimiter=";", first_line=2, columns=(0, 1, 2)
    )
    assert np.all(data["A"] == 1)
    assert np.all(data["B"] == [1, 2, 3])
    assert np.all(data["C"] == [1, 2, 4])


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
        assert fp.readline() == "#,,4.4100003e-07,m\n"
        assert fp.readline() == "#,,4.5e-18,m³\n"
        assert fp.readline() == "#,,4.2971835e-08,mol/L\n"

        assert fp.readline() == "# Median,a,b\n"
        assert fp.readline() == "#,5,9,counts\n"
        assert fp.readline() == "#,,4.5e-19,kg\n"
        assert fp.readline() == "#,,4.413041e-07,m\n"
        assert fp.readline() == "#,,4.5e-18,m³\n"
        assert fp.readline() == "#,,4.2971835e-08,mol/L\n"

        assert fp.readline() == "# Mode,a,b\n"
        assert fp.readline() == "#,5,9.25,counts\n"
        assert fp.readline() == "#,,4.625e-19,kg\n"
        assert fp.readline() == "#,,4.4479151e-07,m\n"
        assert fp.readline() == "#,,4.625e-18,m³\n"
        assert fp.readline() == "#,,4.4165497e-08,mol/L\n"

        assert fp.readline() == "# Limits of detection,a,b\n"
        assert fp.readline() == "#,7.5 - 9.5,8.5,counts\n"
        assert fp.readline() == "#,,4.25e-19,kg\n"
        assert fp.readline() == "#,,4.3297561e-07,m\n"
        assert fp.readline() == "#,,4.25e-18,m³\n"
        assert fp.readline() == "#,,4.058451e-08,mol/L\n"
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


def test_export_singleparticle_results_a_only(tmp_path: Path):
    tmp = tmp_path.joinpath("test_export_results_a_only.csv")

    export_single_particle_results(
        tmp, {"a": results["a"]}, clusters, output_inputs=False, output_arrays=False
    )
    with tmp.open("r") as fp:
        for i in range(5):
            fp.readline()
        assert fp.readline() == "# Detection results,a\n"
        assert fp.readline() == "# Particle number,4\n"
        assert fp.readline() == "# Number error,2\n"
        # Number / Mass concentrations skipped
        fp.readline()
        assert fp.readline() == "# Background,1,counts\n"


def test_export_singleparticle_arrays(tmp_path: Path):
    tmp = tmp_path.joinpath("test_export_arrays.csv")

    export_single_particle_results(
        tmp, results, clusters, times, output_inputs=False, output_results=False
    )

    with tmp.open("r") as fp:
        for i in range(5):
            fp.readline()
        fp.readline()
        assert fp.readline() == "Time,a,b,b,b,b,b\n"
        assert fp.readline() == "s,counts,counts,kg,m,m³,mol/L\n"
        # Todo, compute these
        assert fp.readline() == "0.1,5,,,,,\n"
        assert fp.readline() == "0.2,5,8,4e-19,4.2431377e-07,4e-18,3.8197186e-08\n"
        assert fp.readline() == "0.3,5,9,4.5e-19,4.413041e-07,4.5e-18,4.2971835e-08\n"
        assert fp.readline() == "0.4,5,9,4.5e-19,4.413041e-07,4.5e-18,4.2971835e-08\n"
        assert fp.readline() == "0.5,,10,5e-19,4.5707815e-07,5e-18,4.7746483e-08\n"
        fp.readline()
        assert fp.readline() == "# End of export"

    tmp = tmp_path.joinpath("test_export_arrays_units.csv")
    export_single_particle_results(
        tmp,
        results,
        clusters,
        times,
        output_inputs=False,
        output_results=False,
        units_for_results={"signal": ("cts", 1.0), "mass": ("fg", 1e-18)},
    )
    with tmp.open("r") as fp:
        for i in range(7):
            fp.readline()
        assert fp.readline().startswith("s,cts,cts,fg")
        assert fp.readline().startswith("0.1,5,,,,")


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
        assert fp.readline() == "# Signal,3,0.3663,0.01295,0.6337,0.01295\n"
        assert fp.readline() == ",1,0,0,1,0\n"
        assert fp.readline() == ",1,1,0,0,0\n"
        # No mass / size since only one element
        assert fp.readline() == "# End of export"


def test_export_singleparticle_arrays_with_compositions(tmp_path: Path):
    tmp = tmp_path.joinpath("test_export_arrays_comps.csv")

    export_single_particle_results(
        tmp,
        results,
        clusters,
        times,
        output_inputs=False,
        output_results=False,
        output_compositions=True,
    )

    with tmp.open("r") as fp:
        for i in range(5 + 4):
            fp.readline()
        fp.readline()
        assert fp.readline() == "Time,a,b,b,b,b,b,cluster idx,cluster idx\n"
        assert fp.readline() == "s,counts,counts,kg,m,m³,mol/L,signal,mass\n"
        # Todo, compute these
        assert fp.readline().endswith(",1,1\n")
        assert fp.readline().endswith(",3,2\n")
        assert fp.readline().endswith(",3,2\n")
        assert fp.readline().endswith(",3,1\n")
        assert fp.readline().endswith(",2,1\n")
        fp.readline()
        assert fp.readline() == "# End of export"


def test_export_singleparticle_results_filtered(tmp_path: Path):
    tmp = tmp_path.joinpath("test_export_results_filtered.csv")

    filtered_results = {
        "a": SPCalResult(
            "a.csv",
            np.ones(100),
            np.array([1, 5, 1, 5, 0]),
            np.concatenate((np.zeros(40), np.ones(10), np.zeros(50))),
            SPCalLimit(0.5, np.array([8.0, 10.0]), "Limit", {"kw1": 1.0, "window": 9}),
        ),
        "b": SPCalResult(
            "b.csv",
            np.ones(100),
            np.array([0, 1, 1, 9, 1]),
            np.concatenate((np.zeros(40), np.ones(10), np.zeros(50))),
            SPCalLimit(0.5, np.array([8.0, 10.0]), "Limit", {"kw1": 1.0, "window": 9}),
        ),
    }
    filtered_results["a"]._indicies = np.array([0, 2])
    filtered_results["b"]._indicies = np.array([1, 2, 4])

    export_single_particle_results(
        tmp, filtered_results, clusters, times, output_inputs=False, output_arrays=True
    )
    with tmp.open("r") as fp:
        for i in range(5):
            fp.readline()
        assert fp.readline() == "# Detection results,a,b\n"
        assert fp.readline() == "# Particle number,2,3\n"
        fp.readline()  # number error
        # Number / Mass concentrations skipped
        fp.readline()
        assert fp.readline() == "# Background,1,1,counts\n"
        fp.readline()  # background error

        fp.readline()
        assert fp.readline() == "# Mean,a,b\n"
        assert fp.readline() == "#,1,1,counts\n"

        assert fp.readline() == "# Median,a,b\n"
        assert fp.readline() == "#,1,1,counts\n"

        assert fp.readline() == "# Mode,a,b\n"
        assert fp.readline() == "#,1,1,counts\n"

        fp.readline()  # lod
        fp.readline()  # lod
        fp.readline()
        fp.readline()  # header
        fp.readline()
        fp.readline()
        assert fp.readline() == "0.1,1,\n"
        assert fp.readline() == "0.2,5,1\n"
        assert fp.readline() == "0.3,1,1\n"
        # 0.4,5,9 is filtered
        assert fp.readline() == "0.5,,1\n"
        fp.readline()
        assert fp.readline() == "# End of export"


def test_guess_text_parameters():
    onecol_header = ["Name", "1", "2", "3"]
    delim, skip_rows, columns = guess_text_parameters(onecol_header)
    assert delim == ""
    assert skip_rows == 1
    assert columns == 1


def test_guess_text_parameters_agilent():
    agilent_header = [
        "D:\\Agilent\\ICPMH\\1\\DATA\\Tom\\run.b\\001SMPL.d",
        "Intensity Vs Time,CPS",
        "Acquired    : 00/00/0000 0:00:00 PM using Batch run.b",
        "Time [Sec],S32 -> 48,Gd156 -> 172",
        "0.2312,12274.84,20",
        "0.4402,12304.86,30",
        "0.6492,12114.71,40",
        "0.8582,12244.81,10",
    ]

    delim, skip_rows, columns = guess_text_parameters(agilent_header)
    assert delim == ","
    assert skip_rows == 4
    assert columns == 3

    agilent_header_with_delims = [
        "D:\\Agilent\\ICPMH\\1\\DATA\\Tom\\run,0.1,\tok.b\\001SMPL.d",
        "Intensity Vs Time,CPS",
        "Acquired    : 00/00/0000 0:00:00 PM using Batch run.b",
        "Time [Sec],S32 -> 48,Gd156 -> 172",
        "0.2312,12274.84,20",
        "0.4402,12304.86,30",
        "0.6492,12114.71,40",
        "0.8582,12244.81,10",
    ]

    delim, skip_rows, columns = guess_text_parameters(agilent_header_with_delims)
    assert delim == ","
    assert skip_rows == 4
    assert columns == 3


def test_guess_text_parameters_nu():
    nu_header = [
        "Time (ms),106.905 - seg Full mass spectrum att 1,108.905 - seg Full mass spectrum att 1,196.967 - seg Full mass spectrum att 1",
        "0.09704,0,0,0",
        "0.14556,0,0,0",
        "0.19408,0,0,0",
    ]
    delim, skip_rows, columns = guess_text_parameters(nu_header)
    assert delim == ","
    assert skip_rows == 1
    assert columns == 4


def test_guess_text_parameters_thermo_new_icap():
    icap_header = [
        "sep=,",
        "Number,Time 80Se | 80Se.16O,Intensity (cps) 80Se | 80Se.16O",
        "1,00:00:00.0000500,0",
        "2,00:00:00.0001000,0",
        "3,00:00:00.0001500,0",
        "4,00:00:00.0002000,0",
    ]
    delim, skip_rows, columns = guess_text_parameters(icap_header)
    assert delim == ","
    assert skip_rows == 2
    assert columns == 3


def test_guess_text_parameters_tofwerk():
    tofwerk_header = [
        "Index,timestamp (s),[197Au]+ (cts)",
        "0,0,0",
        "1,0.0009999,0",
        "2,0.0019998,0",
        "3,0.0029997,0",
    ]

    delim, skip_rows, columns = guess_text_parameters(tofwerk_header)
    assert delim == ","
    assert skip_rows == 1
    assert columns == 3
