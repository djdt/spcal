import numpy as np
from pathlib import Path
import pytest
import spcal


def test_accumulate_detections():
    x = np.array([2, 1, 2, 2, 1, 0, 0, 1, 0, 2])
    # Lc == Ld
    sums, labels, regions = spcal.accumulate_detections(x, 1, 1)
    assert np.all(sums == [2, 4, 2])
    assert np.all(labels == [1, 0, 2, 2, 0, 0, 0, 0, 0, 3])
    assert np.all(regions == [[0, 1], [2, 4], [9, 9]])

    # Test regions access
    assert np.all(sums == np.add.reduceat(x, regions.ravel())[::2])

    # Lc < Ld
    sums, labels, regions = spcal.accumulate_detections(x, 0, 1)
    assert np.all(sums == [8, 2])
    assert np.all(labels == [1, 1, 1, 1, 1, 0, 0, 0, 0, 2])
    assert np.all(regions == [[0, 5], [9, 9]])

    # Lc > Ld
    with pytest.raises(ValueError):
        sums, labels, regions = spcal.accumulate_detections(x, 1, 0)

    # Lc > max
    sums, labels, regions = spcal.accumulate_detections(x, 3, 3)
    assert np.all(sums == [])
    assert np.all(labels == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert regions.size == 0

    # Ld > max > Lc
    sums, labels, regions = spcal.accumulate_detections(x, 0, 3)
    assert np.all(sums == [])
    assert np.all(labels == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert regions.size == 0


def test_equations():
    # N = m (kg) * N_A (/mol) / M (kg/mol)
    assert np.all(
        np.isclose(
            spcal.atoms_per_particle(masses=np.array([1.0, 2.0]), molarmass=6.0221e23),
            [1.0, 2.0],
        )
    )
    # c (mol/L) = m (kg) / (V[4.0 / 3.0 * pi * (d (m) / 2) ^ 3] (m^3) * 1000 (L/m^3)) / M (kg/mol)
    assert np.all(
        np.isclose(
            spcal.cell_concentration(
                np.array([2.0 * np.pi, 4.0 * np.pi]), diameter=2e-3, molarmass=1e4
            ),
            [150.0, 300.0],
        )
    )
    # η = (m (kg) * N) / (c (kg/L) * V (L/s) * t (s))
    assert np.isclose(
        spcal.nebulisation_efficiency_from_concentration(
            count=10, mass=80.0, concentration=10.0, flowrate=20.0, time=4.0
        ),
        1.0,
    )
    # η = (m (kg) * s (L/kg) * f) / (I * t (s) * V (L/s))
    assert np.all(
        np.isclose(  # sensitive to mean value
            spcal.nebulisation_efficiency_from_mass(
                signal=np.array([10.0, 20.0, 30.0]),
                mass=10.0,
                response_factor=20.0,
                mass_fraction=0.5,
                dwell=10.0,
                flowrate=2.0,
            ),
            0.25,
        ),
    )
    # m (kg) = (η * t (s) * I * V (L/s)) / (s (L/kg) * f)
    assert np.all(
        np.isclose(
            spcal.particle_mass(
                signal=np.array([1.0, 2.0, 3.0]),
                efficiency=0.5,
                dwell=0.5,
                flowrate=4.0,
                response_factor=2.0,
                mass_fraction=0.5,
            ),
            np.array([1.0, 2.0, 3.0]),
        )
    )
    # PNC (/L) = N / (η * V (L/s) * T (s))
    assert np.isclose(
        spcal.particle_number_concentration(
            1000, efficiency=0.2, flowrate=0.1, time=50.0
        ),
        1000.0,
    )
    # d (m) = cbrt((6.0 * m (kg)) / (π * ρ (kg/m3)) )
    assert np.all(
        np.isclose(
            spcal.particle_size(
                masses=np.array([np.pi / 60.0, np.pi / 480.0]), density=0.1
            ),
            np.array([1.0, 0.5]),
        )
    )
    # C (kg/L) = sum(m (kg)) / (η * V (L/s) * T (s))
    assert np.isclose(
        spcal.particle_total_concentration(
            np.array([0.1, 0.2, 0.3, 0.4]), efficiency=0.1, flowrate=2.0, time=5.0
        ),
        1.0,
    )

    # m (kg) = 4.0 / (3.0 * pi) * (d (m) / 2) ^ 3 * ρ (kg/m3)
    assert np.isclose(
        spcal.reference_particle_mass(diameter=0.2, density=750.0 / np.pi), 1.0
    )

    # Test that particle mass is recoverable from same reference mass
    kws = dict(
        signal=4.3, response_factor=7.8, mass_fraction=0.91, dwell=0.87, flowrate=0.65
    )
    assert np.isclose(
        spcal.particle_mass(
            efficiency=spcal.nebulisation_efficiency_from_mass(mass=5.6, **kws), **kws
        ),
        5.6,
    )


def test_determined_sizes():
    data = np.load(Path(__file__).parent.joinpath("data/agilent_au_data.npz"))

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
        # 15 nm
        yc, yd = spcal.poisson_limits(ub, epsilon=0.5)
        detections, _, _ = spcal.accumulate_detections(x, yc + ub, yd + ub)

        masses = spcal.particle_mass(
            detections,
            dwell=dwelltime,
            efficiency=efficiency,
            flowrate=uptake,
            response_factor=response,
            mass_fraction=1.0,
        )

        sizes = spcal.particle_size(masses, density=density)

        assert np.isclose(
            np.mean(sizes) * 1e9, expected, rtol=0.01, atol=0
        )  # within 1%
        assert np.isclose(
            np.median(sizes) * 1e9, expected, rtol=0, atol=0.5
        )  # within 0.5 nm
