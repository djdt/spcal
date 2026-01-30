import numpy as np

from pathlib import Path
from spcal.lib.spcalext import extraction


def test_extract_cpln_parameters():
    data = np.load(Path(__file__).parent.joinpath("data/cpln_simulations.npz"))
    mask = np.ones((1000000, 1), dtype=bool)
    for file in data:
        params = [float(x) for x in file.split(",")]
        params = extraction.extract_cpln_parameters(np.atleast_2d(data[file]).T, mask)
        assert np.allclose(params, [params], rtol=0.01)
