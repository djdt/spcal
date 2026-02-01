from spcal import datafile
from pathlib import Path
import numpy as np


def test_spcal_datafile():
    df = datafile.SPCalDataFile(Path(), np.linspace(0, 1, 101), "quadrupole")

    assert df.event_time == 0.01
    assert df.total_time == 1.0
    assert not df.isTOF()


def test_spcal_datafile_text():
    pass


def test_spcal_datafile_nu():
    pass


def test_spcal_datafile_tofwerk():
    pass
