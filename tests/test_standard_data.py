from pathlib import Path
import numpy as np
from spcal import particle, detection, poisson


def test_standard_sizes():
    data = np.load(Path(__file__).parent.joinpath("data/agilent_au_data.npz"))
    # Data for 15 and 50 nm standards,
    # Reported error is approx. +- 10 %, test is 1% and 0.5 nm from mean and median

    # Determined experimentally
    uptake = 1.567e-6
    dwelltime = 1e-4
    response = 16.08e9
    efficiency = 0.062
    density = 19.32e3

    for x, expected in [
        (data["au15nm"], 15.0),
        (data["au50nm"], 50.0),
    ]:
        ub = np.mean(x)
        assert isinstance(ub, float)
        # 15 nm
        yc, yd = poisson.formula_c(ub)
        detections, _, _ = detection.accumulate_detections(x, yc + ub, yd + ub)

        masses = particle.particle_mass(
            detections,
            dwell=dwelltime,
            efficiency=efficiency,
            flow_rate=uptake,
            response_factor=response,
            mass_fraction=1.0,
        )

        sizes = particle.particle_size(masses, density=density)

        assert np.isclose(
            np.mean(sizes) * 1e9, expected, rtol=0.01, atol=0.0
        )  # within 1%
        assert np.isclose(
            np.median(sizes) * 1e9, expected, rtol=0.0, atol=1.0
        )  # within 1.0 nm
