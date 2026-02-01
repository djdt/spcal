from pathlib import Path

import numpy as np

from spcal.io.text import (
    guess_text_parameters,
    is_text_file,
    read_single_particle_file,
)


def test_io_is_text_file():
    assert is_text_file(Path(__file__).parent.joinpath("data/text/text_normal.csv"))
    assert not is_text_file(Path(__file__).parent.joinpath("data/text/text_normal.bad"))
    assert not is_text_file(Path(__file__).parent.joinpath("data/text/fake.txt"))


def test_io_text_import():
    path = Path(__file__).parent.joinpath("data/text/text_normal.csv")
    data = read_single_particle_file(path, skip_rows=2)
    assert np.all(data["A"] == 1)
    assert np.all(data["B"] == [1, 2, 3])
    assert np.all(data["C"] == [1, 2, 4])


def test_io_text_import_euro():
    path = Path(__file__).parent.joinpath("data/text/text_euro.csv")
    data = read_single_particle_file(path, delimiter=";", skip_rows=2)
    assert np.all(data["A"] == 1)
    assert np.all(data["B"] == [1, 2, 3])
    assert np.all(data["C"] == [1, 2, 4])


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
