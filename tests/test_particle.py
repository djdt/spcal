import numpy as np

from spcal import particle


def test_equations():
    # N = m (kg) * N_A (/mol) / M (kg/mol)
    assert np.all(
        np.isclose(
            particle.atoms_per_particle(masses=np.array([1.0, 2.0]), molar_mass=6.0221e23),
            [1.0, 2.0],
        )
    )
    # c (mol/L) = m (kg) / (V[4.0 / 3.0 * pi * (d (m) / 2) ^ 3] (m^3) * 1000 (L/m^3)) / M (kg/mol)
    assert np.all(
        np.isclose(
            particle.cell_concentration(
                np.array([2.0 * np.pi, 4.0 * np.pi]), diameter=2e-3, molar_mass=1e4
            ),
            [150.0, 300.0],
        )
    )
    # η = (m (kg) * N) / (c (kg/L) * V (L/s) * t (s))
    assert np.isclose(
        particle.nebulisation_efficiency_from_concentration(
            count=10, mass=80.0, concentration=10.0, flow_rate=20.0, time=4.0
        ),
        1.0,
    )
    # η = (m (kg) * s (L/kg) * f) / (I * t (s) * V (L/s))
    assert np.all(
        np.isclose(  # sensitive to mean value
            particle.nebulisation_efficiency_from_mass(
                signal=np.array([10.0, 20.0, 30.0]),
                mass=10.0,
                response_factor=20.0,
                mass_fraction=0.5,
                dwell=10.0,
                flow_rate=2.0,
            ),
            0.25,
        ),
    )
    # m (kg) = (η * t (s) * I * V (L/s)) / (s (L/kg) * f)
    assert np.all(
        np.isclose(
            particle.particle_mass(
                signal=np.array([1.0, 2.0, 3.0]),
                efficiency=0.5,
                dwell=0.5,
                flow_rate=4.0,
                response_factor=2.0,
                mass_fraction=0.5,
            ),
            np.array([1.0, 2.0, 3.0]),
        )
    )
    # PNC (/L) = N / (η * V (L/s) * T (s))
    assert np.isclose(
        particle.particle_number_concentration(
            1000, efficiency=0.2, flow_rate=0.1, time=50.0
        ),
        1000.0,
    )
    # d (m) = cbrt((6.0 * m (kg)) / (π * ρ (kg/m3)) )
    assert np.all(
        np.isclose(
            particle.particle_size(
                masses=np.array([np.pi / 60.0, np.pi / 480.0]), density=0.1
            ),
            np.array([1.0, 0.5]),
        )
    )
    # C (kg/L) = sum(m (kg)) / (η * V (L/s) * T (s))
    assert np.isclose(
        particle.particle_total_concentration(
            np.array([0.1, 0.2, 0.3, 0.4]), efficiency=0.1, flow_rate=2.0, time=5.0
        ),
        1.0,
    )

    # m (kg) = 4.0 / (3.0 * pi) * (d (m) / 2) ^ 3 * ρ (kg/m3)
    assert np.isclose(
        particle.reference_particle_mass(diameter=0.2, density=750.0 / np.pi), 1.0
    )

    # Test that particle mass is recoverable from same reference mass
    kws = dict(
        signal=4.3, response_factor=7.8, mass_fraction=0.91, dwell=0.87, flow_rate=0.65
    )
    assert np.isclose(
        particle.particle_mass(
            efficiency=particle.nebulisation_efficiency_from_mass(mass=5.6, **kws), **kws
        ),
        5.6,
    )
