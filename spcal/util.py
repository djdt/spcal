import numpy as np

from typing import Tuple, Union


def accumulate_detections(
    y: np.ndarray,
    limit_accumulation: Union[float, np.ndarray],
    limit_detection: Union[float, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns an array of accumulated detections.

    Contiguous regions above `limit_accumulation` that contain at least one value above
    `limit_detection` are summed.

    Args:
        y: array
        limit_detection: value(s) for detection of region
        limit_accumulation: minimum accumulation value(s)

    Returns:
        summed detection regions
        labels of regions
        regions [starts, ends]
    """
    if np.any(limit_detection < limit_accumulation):
        raise ValueError("limit_detection must be greater than limit_accumulation.")
    # Get start and end positions of regions above accumulation limit
    diff = np.diff((y > limit_accumulation).astype(np.int8), prepend=0)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    # Stack into pairs of start, end. If no final end position set it as end of array.
    end_point_added = False
    if starts.size != ends.size:
        ends = np.concatenate((ends, [diff.size - 1]))  # -1 for reduceat
        end_point_added = True
    regions = np.stack((starts, ends), axis=1)

    # Get maximum values in each region
    detections = np.logical_or.reduceat(y > limit_detection, regions.ravel())[::2]
    # Remove regions without a max value above detection limit
    regions = regions[detections]
    # Sum regions
    sums = np.add.reduceat(y, regions.ravel())[::2]

    # Create a label array of detections
    labels = np.zeros(y.size, dtype=np.int16)
    ix = np.arange(1, regions.shape[0] + 1)
    # Set start, end pairs to +i, -i
    labels[regions[:, 0]] = ix
    if end_point_added:
        labels[regions[:-1, 1]] = -ix[:-1]
    else:
        labels[regions[:, 1]] = -ix
    # Cumsum to label
    labels = np.cumsum(labels)

    return sums, labels, regions


def poisson_limits(
    ub: np.ndarray,
    epsilon: float = 0.5,
    force_epsilon: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calulate Yc and Yd for mean `ub`.

    If `ub` if lower than 5.0, the correction factor `epsilon` is added to `ub`.
    Lc and Ld can be calculated by adding `ub` to `Yc` and `Yd`.

    Args:
        ub: mean of background
        epsilon: low `ub` correct factor
        force_epsilon: always use `epsilon`

    Returns:
        Yc, gross count critical value
        Yd, gross count detection limit

    References:
        Currie, L. A. (1968). Limits for qualitative detection and quantitative
            determination. Application to radiochemistry.
            Analytical Chemistry, 40(3), 586–593.
            doi:10.1021/ac60259a007
        Currie, L.A. On the detection of rare, and moderately rare, nuclear events.
            J Radioanal Nucl Chem 276, 285–297 (2008).
            https://doi.org/10.1007/s10967-008-0501-5
    """
    # 5 counts limit to maintain 0.05 alpha / beta (Currie 2008)
    if force_epsilon:
        ub = ub + epsilon
    else:
        ub = np.where(ub < 5.0, ub + epsilon, ub)
    # Yc and Yd for paired distribution (Currie 1969)
    return 2.33 * np.sqrt(ub), 2.71 + 4.65 * np.sqrt(ub)


# Particle functions


def atoms_per_particle(
    masses: Union[float, np.ndarray],
    molarmass: float,
) -> Union[float, np.ndarray]:
    """Number of atoms per particle.
    N = m (kg) * N_A (/mol) / M (kg/mol)

    Args:
        masses: array of particle masses (kg)
        molarmass: molecular weight (kg/mol)
    """
    Na = 6.02214076e23
    return masses * Na / molarmass


def cell_concentration(
    masses: Union[float, np.ndarray], diameter: float, molarmass: float
) -> Union[float, np.ndarray]:
    """Calculates intracellular concentrations.
    c (mol/L) = m (kg) / V[4.0 / 3.0 * pi * (d (m) / 2) ^ 3] (m^3) * 1000 (L/m^3) / M (kg/mol)

    Args:
        masses: array of material masses (kg)
        diameter: cell diameter (m)
        molarmass: molecular weight (kg/mol)
    """
    return masses / ((4.0 / 3.0 * np.pi * (diameter / 2.0) ** 3) * 1000.0 * molarmass)


def nebulisation_efficiency_from_concentration(
    count: int, concentration: float, mass: float, flowrate: float, time: float
) -> float:
    """The nebulistaion efficiency given a defined concentration.
    η = (m (kg) * N) / (c (kg/L) * V (L/s) * t (s))

    Args:
        count: number of detected particles
        concentration: of reference material (kg/L)
        mass: of reference material (kg)
        flowrate: sample inlet flow (L/s)
        time: total aquisition time (s)
    """

    return (mass * (count * flowrate * time)) / concentration


def nebulisation_efficiency_from_mass(
    signal: Union[float, np.ndarray],
    dwell: float,
    mass: float,
    flowrate: float,
    response_factor: float,
    mass_fraction: float = 1.0,
) -> float:
    """Calculates efficiency for signals given a defined mass.
    η = (m (kg) * s (L/kg)) / (I * f * t (s) * V (L/s))

    Args:
        signal: array of reference particle signals
        dwell: dwell time (s)
        mass: of reference particle (kg)
        flowrate: sample inlet flowrate (L/s)
        response_factor: counts / concentration (kg/L)
        mass_fraction: molar mass analyte / molar mass particle
    """
    signal = np.mean(signal)
    return (mass * response_factor) / (signal * (dwell * flowrate * mass_fraction))


def particle_mass(
    signal: Union[float, np.ndarray],
    dwell: float,
    efficiency: float,
    flowrate: float,
    response_factor: float,
    mass_fraction: float = 1.0,
) -> Union[float, np.ndarray]:
    """Array of particle masses given their integrated responses.
    m (kg) = (η * t (s) * I * V (L/s)) / (s (L/kg) * f)

    Args:
        signal: array of particle signals
        dwell: dwell time (s)
        efficiency: nebulisation efficiency
        flowrate: sample inlet flowrate (L/s)
        response_factor: counts / concentration (kg/L)
        mass_fraction:  molar mass analyte / molar mass particle
    """
    return signal * (dwell * flowrate * efficiency / (response_factor * mass_fraction))


def particle_number_concentration(
    count: int, efficiency: float, flowrate: float, time: float
) -> float:
    """Number concentration of particles.
    PNC (/L) = N / (η * V (L/s) * T (s))

    Args:
        count: number of detected particles
        efficiency: nebulisation efficiency
        flowrate: sample inlet flowrate (L/s)
        time: total aquisition time (s)
    """
    return count / (efficiency * flowrate * time)


def particle_size(
    masses: Union[float, np.ndarray], density: float
) -> Union[float, np.ndarray]:
    """Array of particle diameters.
    d (m) = cbrt((6.0 * m (kg)) / (π * ρ (kg/m3)))

    Args:
        masses: array of particle signals (kg)
        density: reference density (kg/m3)
    """
    return np.cbrt(6.0 / (np.pi * density) * masses)


def particle_total_concentration(
    masses: Union[float, np.ndarray], efficiency: float, flowrate: float, time: float
) -> float:
    """Concentration of material.
    C (kg/L) = sum(m (kg)) / (η * V (L/s) * T (s))

    Args:
        masses: array of particle signals (kg)
        efficiency: nebulisation efficiency
        flowrate: sample inlet flowrate (L/s)
        time: total aquisition time (s)
    """

    return np.sum(masses) / (efficiency * flowrate * time)


def reference_particle_mass(density: float, diameter: float) -> float:
    """Calculates particle mass assusming a spherical particle.
    m (kg) = 4 / 3 * pi * (d (m) / 2) ^ 3 * ρ (kg/m3)

    Args:
        density: reference density (kg/m3)
        diameter: reference diameter (m)
    """
    return 4.0 / 3.0 * np.pi * (diameter / 2.0) ** 3 * density


# def reference_particle_size(mass_std: float, density_std: float) -> float:
#     """Calculates particle diameter in m.

#     Args:
#         mass: particle mass (kg)
#         density: reference density (kg/m3)
#     """
#     return np.cbrt(6.0 / np.pi * mass_std / density_std)
