"""Functions for particle calculations."""
import numpy as np


def atoms_per_particle(
    masses: float | np.ndarray,
    molar_mass: float,
) -> float | np.ndarray:
    """Number of atoms per particle.
    N = m (kg) * N_A (/mol) / M (kg/mol)

    Args:
        masses: array of particle masses (kg)
        molar_mass: molecular weight (kg/mol)
    """
    Na = 6.02214076e23
    return masses * Na / molar_mass


def cell_concentration(
    masses: float | np.ndarray, diameter: float, molar_mass: float
) -> float | np.ndarray:
    """Calculates intracellular concentrations.
    c (mol/L) = m (kg) / (V[4.0 / 3.0 * pi * (d (m) / 2) ^ 3] (m^3) * 1000 (L/m^3))
        / M (kg/mol)

    Args:
        masses: array of material masses (kg)
        diameter: cell diameter (m)
        molar_mass: molecular weight (kg/mol)
    """
    return masses / ((4.0 / 3.0 * np.pi * (diameter / 2.0) ** 3) * 1000.0 * molar_mass)


def nebulisation_efficiency_from_concentration(
    count: int, concentration: float, mass: float, flow_rate: float, time: float
) -> float:
    """The nebulistaion efficiency given a defined concentration.
    η = (m (kg) * N) / (c (kg/L) * V (L/s) * t (s))

    Args:
        count: number of detected particles
        concentration: of reference material (kg/L)
        mass: of reference material (kg)
        flow_rate: sample inlet flow (L/s)
        time: total aquisition time (s)
    """

    return (mass * count) / (flow_rate * time * concentration)


def nebulisation_efficiency_from_mass(
    signal: float | np.ndarray,
    dwell: float,
    mass: float,
    flow_rate: float,
    response_factor: float,
    mass_fraction: float = 1.0,
) -> float:
    """Calculates efficiency for signals given a defined mass.
    η = (m (kg) * s (L/kg) * f) / (I * t (s) * V (L/s))

    Args:
        signal: array of reference particle signals
        dwell: dwell time (s)
        mass: of reference particle (kg)
        flow_rate: sample inlet flowrate (L/s)
        response_factor: counts / concentration (kg/L)
        mass_fraction: molar mass analyte / molar mass particle
    """
    signal = np.mean(signal)
    return (mass * response_factor * mass_fraction) / (signal * (dwell * flow_rate))


def particle_mass(
    signal: float | np.ndarray,
    dwell: float,
    efficiency: float,
    flow_rate: float,
    response_factor: float,
    mass_fraction: float = 1.0,
) -> float | np.ndarray:
    """Array of particle masses given their integrated responses.
    m (kg) = (η * t (s) * I * V (L/s)) / (s (L/kg) * f)

    Args:
        signal: array of particle signals
        dwell: dwell time (s)
        efficiency: nebulisation efficiency
        flow_rate: sample inlet flowrate (L/s)
        response_factor: counts / concentration (kg/L)
        mass_fraction:  molar mass analyte / molar mass particle
    """
    return signal * (dwell * flow_rate * efficiency / (response_factor * mass_fraction))


def particle_number_concentration(
    count: int, efficiency: float, flow_rate: float, time: float
) -> float:
    """Number concentration.
    PNC (/L) = N / (η * V (L/s) * T (s))

    Args:
        count: number of detected particles
        efficiency: nebulisation efficiency
        flow_rate: sample inlet flowrate (L/s)
        time: total aquisition time (s)
    """
    return count / (efficiency * flow_rate * time)


def particle_size(masses: float | np.ndarray, density: float) -> float | np.ndarray:
    """Array of particle diameters.
    d (m) = cbrt((6.0 * m (kg)) / (π * ρ (kg/m3)))

    Args:
        masses: array of particle signals (kg)
        density: reference density (kg/m3)
    """
    return np.cbrt(6.0 / (np.pi * density) * masses)


def particle_total_concentration(
    masses: float | np.ndarray, efficiency: float, flow_rate: float, time: float
) -> float:
    """Concentration of material.
    C (kg/L) = sum(m (kg)) / (η * V (L/s) * T (s))

    Args:
        masses: array of particle signals (kg)
        efficiency: nebulisation efficiency
        flow_rate: sample inlet flowrate (L/s)
        time: total aquisition time (s)
    """

    return np.sum(masses) / (efficiency * flow_rate * time)


def reference_particle_mass(density: float, diameter: float) -> float:
    """Calculates particle mass assusming a spherical particle.
    m (kg) = 4 / 3 * pi * (d (m) / 2) ^ 3 * ρ (kg/m3)

    Args:
        density: reference density (kg/m3)
        diameter: reference diameter (m)
    """
    return 4.0 / 3.0 * np.pi * (diameter / 2.0) ** 3 * density
