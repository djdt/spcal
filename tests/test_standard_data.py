from pathlib import Path

import numpy as np

from spcal import cluster, detection, particle, poisson


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
        (data["au15nm"], 15.0e-9),
        (data["au50nm"], 50.0e-9),
    ]:
        ub = np.mean(x)
        assert isinstance(ub, float)
        # 15 nm
        yc, _ = poisson.formula_c(ub, alpha=0.05)
        detections, _, _ = detection.accumulate_detections(x, ub, yc + ub)

        masses = particle.particle_mass(
            detections,
            dwell=dwelltime,
            efficiency=efficiency,
            flow_rate=uptake,
            response_factor=response,
            mass_fraction=1.0,
        )

        sizes = particle.particle_size(masses, density=density)
        # within 1.0 nm
        assert np.isclose(np.mean(sizes), expected, rtol=0.0, atol=1e-9)
        assert np.isclose(np.median(sizes), expected, rtol=0.0, atol=1e-9)


def test_standard_compositions():
    """Test data from DOI: 10.1039/d2ja00116k and supp info"""

    npz = np.load(Path(__file__).parent.joinpath("data/compositions.npz"))

    molar_mass = {
        "Fe": 55.845,
        "Co": 58.9332,
        "Ni": 58.6934,
        "Zn": 65.39,
        "Ag": 107.8682,
        "Au": 196.9665,
    }
    response_cps = {
        "Fe": 35492.9,
        "Co": 48377.6,
        "Ni": 10444.3,
        "Zn": 5465.4,
        "Ag": 14388.6,
        "Au": 16436.0,
    }
    theoretical_fractions = {
        "60nm AuAg": (0.90, 0.10),  # Ag, Au
        "80nm AuAg": (0.83, 0.17),  # Ag, Au
        "FeCoZn": (0.67, 0.17, 0.17),
    }

    for file, value in theoretical_fractions.items():
        data = npz[file]
        for name in data.dtype.names:
            data[name] = data[name] / response_cps[name] / molar_mass[name]
        X = cluster.prepare_data_for_clustering(data)
        X = X[np.all(X != 0, axis=1)]  # Remove particles with all elements
        means, stds, _ = cluster.agglomerative_cluster(X, 0.03)

        # mean +- std contains theoretical value
        for a, s, b in zip(means[0], stds[0], value):
            assert abs(a - b) < s
