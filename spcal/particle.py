"""Functions for particle calculations."""

import numpy as np


def atoms_per_particle(
    masses: float | np.ndarray,
    molar_mass: float,
) -> float | np.ndarray:
    """Number of atoms per particle.

    :math:`N = \\frac{m (kg) N_A ({mol}^{-1})}{M ({kg} \\cdot {mol}^{-1})}`

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

    :math:`c (mol \\cdot L^{-1}) = \\frac{6 m (kg)}{
    4 \\pi d (m)^3 M ({kg} \\cdot {mol}^{-1}) 1000 (L \\cdot m^3)}`

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

    :math:`\\eta = \\frac{m (kg) N}{c ({kg} \\cdot L^{-1}) V (L \\cdot s^{-1}) t (s)}`

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

    :math:`\\eta = \\frac{m (kg) s (L \\cdot {kg}^{-1}) f}
    {I V (L \\cdot {s}^{-1}) t (s)}`

    Args:
        signal: array of reference particle signals
        dwell: dwell time (s)
        mass: of reference particle (kg)
        flow_rate: sample inlet flowrate (L/s)
        response_factor: counts / concentration (kg/L)
        mass_fraction: molar mass analyte / molar mass particle
    """
    if isinstance(signal, np.ndarray):
        signal = signal[signal > 0]  # filter zeros and nan
    signal = float(np.mean(signal))
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

    :math:`m (kg) = \\frac{\\eta t (s) I V (L \\cdot s^{-1})}
    {s (L \\cdot {kg}^{-1}) f}`

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

    :math:`{PNC} (L^{-1}) = \\frac{N}{\\eta V (L \\cdot s^{-1}) T (s)}`

    Args:
        count: number of detected particles
        efficiency: nebulisation efficiency
        flow_rate: sample inlet flowrate (L/s)
        time: total aquisition time (s)
    """
    return count / (efficiency * flow_rate * time)


def particle_size(masses: float | np.ndarray, density: float) -> float | np.ndarray:
    """Array of particle diameters.

    :math:`d (m) = \\sqrt[3]{\\frac{6 m (kg)}{\\pi \\rho ({kg} \\cdot m^3)}}`

    Args:
        masses: array of particle signals (kg)
        density: reference density (kg/m3)
    """
    return np.cbrt(6.0 / (np.pi * density) * masses)


def particle_total_concentration(
    masses: float | np.ndarray, efficiency: float, flow_rate: float, time: float
) -> float:
    """Concentration of material.

    :math:`C (kg L^{-1}) = \\sum{\\frac{m (kg)}{\\eta V (L \\cdot s^{-1}) T (s)}}`

    Args:
        masses: array of particle signals (kg)
        efficiency: nebulisation efficiency
        flow_rate: sample inlet flowrate (L/s)
        time: total aquisition time (s)
    """

    return np.sum(masses) / (efficiency * flow_rate * time)


def reference_particle_mass(density: float, diameter: float) -> float:
    """Calculates particle mass assusming a spherical particle.

    :math:`m (kg) = \\frac{4}{3} \\pi (\\frac{d (m)}{2})^3 \\rho ({kg} \\cdot m^3)`

    Args:
        density: reference density (kg/m3)
        diameter: reference diameter (m)
    """
    return 4.0 / 3.0 * np.pi * (diameter / 2.0) ** 3 * density
