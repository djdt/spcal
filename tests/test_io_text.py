import pytest
from pathlib import Path

import numpy as np

from spcal.io.text import (
    guess_text_parameters,
    is_text_file,
    read_single_particle_file,
    guess_event_time,
)

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
nu_header = [
    "Time (ms),106.905 - seg Full mass spectrum att 1,108.905 - seg Full mass spectrum att 1,196.967 - seg Full mass spectrum att 1",
    "0.09704,0,0,0",
    "0.14556,0,0,0",
    "0.19408,0,0,0",
]
icap_header = [
    "sep=,",
    "Number,Time 80Se | 80Se.16O,Intensity (cps) 80Se | 80Se.16O",
    "1,00:00:00.0000500,0",
    "2,00:00:00.0001000,0",
    "3,00:00:00.0001500,0",
    "4,00:00:00.0002000,0",
]
single_column_header = ["Au197", "1", "0", "0", "3"]
tofwerk_header = [
    "Index,timestamp (s),[197Au]+ (cts)",
    "0,0,0",
    "1,0.0009999,0",
    "2,0.0019998,0",
    "3,0.0029997,0",
]


def test_io_is_text_file(test_data_path: Path):
    assert is_text_file(test_data_path.joinpath("text/text_normal.csv"))
    assert not is_text_file(test_data_path.joinpath("text/text_normal.bad"))
    assert not is_text_file(test_data_path.joinpath("text/fake.txt"))


def test_io_text_import(test_data_path: Path):
    path = test_data_path.joinpath("text/text_normal.csv")
    data = read_single_particle_file(path, skip_rows=2)
    assert np.all(data["A"] == 1)
    assert np.all(data["B"] == [1, 2, 3])
    assert np.all(data["C"] == [1, 2, 4])


def test_io_text_import_euro(test_data_path: Path):
    path = test_data_path.joinpath("text/text_euro.csv")
    data = read_single_particle_file(path, delimiter=";", skip_rows=2)
    assert np.all(data["A"] == 1)
    assert np.all(data["B"] == [1, 2, 3])
    assert np.all(data["C"] == [1, 2, 4])


def test_io_text_import_onecol(test_data_path: Path):
    path = test_data_path.joinpath("text/text_onecol.csv")
    data = read_single_particle_file(path, delimiter=";", skip_rows=1)
    assert np.all(data["A"] == [1, 2, 3, 4, 5])


def test_guess_text_parameters():
    onecol_header = ["Name", "1", "2", "3"]
    delim, skip_rows, columns = guess_text_parameters(onecol_header)
    assert delim == ""
    assert skip_rows == 1
    assert columns == 1


def test_io_text_import_agilent(test_data_path: Path):
    path = test_data_path.joinpath("text/agilent_au50nm.csv")
    data = read_single_particle_file(path, delimiter=",", skip_rows=4)
    assert np.all(np.isfinite(data["Au197"]))
    assert np.isclose(data["Au197"].mean(), 6.2062)


def test_io_text_import_perkin_elmer(test_data_path: Path):
    path = test_data_path.joinpath("text/perkin_elmer.csv")
    data = read_single_particle_file(path, delimiter=",", skip_rows=1)
    assert np.all(data["Au"] == np.arange(10))


def test_io_text_import_new_icap(test_data_path: Path):
    path = test_data_path.joinpath("text/thermo_icap_export.csv")
    data = read_single_particle_file(path, delimiter=",", skip_rows=2)
    assert data.dtype.names is not None
    assert data["Time_80Se_|_80Se.16O"].dtype == np.float32
    assert np.all(~np.isnan(data["Time_80Se_|_80Se.16O"]))  # converted correctly


def test_guess_text_parameters_agilent():
    delim, skip_rows, columns = guess_text_parameters(agilent_header)
    assert delim == ","
    assert skip_rows == 4
    assert columns == 3

    delim, skip_rows, columns = guess_text_parameters(agilent_header_with_delims)
    assert delim == ","
    assert skip_rows == 4
    assert columns == 3


def test_guess_text_parameters_nu():
    delim, skip_rows, columns = guess_text_parameters(nu_header)
    assert delim == ","
    assert skip_rows == 1
    assert columns == 4


def test_guess_text_parameters_thermo_new_icap():
    delim, skip_rows, columns = guess_text_parameters(icap_header)
    assert delim == ","
    assert skip_rows == 2
    assert columns == 3


def test_guess_text_parameters_single_column():
    delim, skip_rows, columns = guess_text_parameters(single_column_header)
    assert delim == ""
    assert skip_rows == 1
    assert columns == 1


def test_guess_text_parameters_tofwerk():
    delim, skip_rows, columns = guess_text_parameters(tofwerk_header)
    assert delim == ","
    assert skip_rows == 1
    assert columns == 3


def test_guess_event_time():
    lines = ["time (s);data", "0.1;1.0", "0.2;2.0", "0.3;3.0", "0.4;2.0"]
    for test_unit in ["s", "ms", "µs", "ns"]:
        lines[0] = f"time ({test_unit});data"
        val, unit = guess_event_time(lines, ";", 1)
        assert val == 0.1
        assert unit == test_unit

    lines[0] = "time;data"
    val, unit = guess_event_time(lines, ";", 1)
    assert val == 0.1
    assert unit is None

    lines[0] = "index;data"
    with pytest.raises(StopIteration):
        guess_event_time(lines, ";", 1)


def test_guess_event_time_agilent():
    val, unit = guess_event_time(agilent_header, ",", skip_rows=4)
    assert val == 0.209
    assert unit == "s"

    val, unit = guess_event_time(agilent_header_with_delims, ",", skip_rows=4)
    assert val == 0.209
    assert unit == "s"


def test_guess_event_time_nu():
    val, unit = guess_event_time(nu_header, ",", skip_rows=1)
    assert val == 0.04852
    assert unit == "ms"


def test_guess_event_time_thermo_new_icap():
    val, unit = guess_event_time(icap_header, ",", skip_rows=2)
    assert np.isclose(val, 5e-5)
    assert unit is None


def test_guess_event_time_single_column():
    with pytest.raises(StopIteration):
        guess_event_time(single_column_header, ",", skip_rows=1)


def test_guess_event_time_tofwerk():
    val, unit = guess_event_time(tofwerk_header, ",", skip_rows=1)
    assert val == 0.0009999
    assert unit == "s"
