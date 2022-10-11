import numpy as np

from typing import Dict, Optional, Tuple, Union


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
        # -1 for reduceat
        ends = np.concatenate((ends, [diff.size - 1]))  # type: ignore
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


def detection_maxima(y: np.ndarray, regions: np.ndarray) -> np.ndarray:
    """Calculates the maxima of each region.

    Args:
        y: array
        regions: regions from `accumulate_detections`

    Returns:
        idx of maxima
    """
    # The width of each detection region
    widths = regions[:, 1] - regions[:, 0]  # type: ignore
    # peak indicies for max width
    indicies = regions[:, 0] + np.arange(np.amax(widths) + 1)[:, None]
    indicies = np.clip(
        indicies, regions[0, 0], regions[-1, 1] - 1
    )  # limit to first to last region
    # limit to peak width
    indicies = np.where(indicies - regions[:, 0] < widths, indicies, regions[:, 1])
    # return indcies that is at maxima
    return np.argmax(y[indicies], axis=0) + regions[:, 0]


def detection_element_fractions(
    sums: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray],
    regions: Dict[str, np.ndarray],
) -> np.ndarray:
    """Computes the relative fraction of each element in each detection.
    Recalculates the start and end point of each peak from *all* element data.
    Regions that overlap will be combined into a single region. Fractions are calculated
    as the sum of all regions contained within each of the reclaculated regions.
    Each argument must have the same dictionaty keys.

    Args:
        sums: dict of detection counts, sizes, mass, ...
        labels: dict of labels from `accumulate_detections`
        regions: dict of regions from `accumulate_detections`

    Returns:
        dict of sum fraction per peak

    """
    if sums.keys() != labels.keys() or sums.keys() != regions.keys():
        raise ValueError(
            "detection_element_fractions: sums, labels and regions must have the same keys."
        )
    names = list(sums.keys())

    # Get regions from all elements
    # Some regions may overlap, these will be combined
    any_label = np.zeros(labels[names[0]].size, dtype=np.int8)
    for name in names:
        any_label[labels[name] > 0] = 1

    # Caclulate start and end points of each region
    diff = np.diff(any_label, prepend=0)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)

    # Stack into pairs of start, end. If no final end position set it as end of array.
    if starts.size != ends.size:
        ends = np.concatenate((ends, [diff.size - 1]))  # type: ignore
    all_regions = np.stack((starts, ends), axis=1)

    # Init empty
    fractions = np.empty(starts.size, dtype=[(n, np.float64) for n in names])
    total = np.zeros(starts.size, dtype=float)
    for name in names:
        # Positions in name's region that corresponds to the combined regions
        idx = (regions[name][:, 0] >= all_regions[:, 0, None]) & (
            regions[name][:, 1] <= all_regions[:, 1, None]
        )
        # Compute the total signal in the region
        fractions[name] = np.sum(np.where(idx, sums[name], 0), axis=1)
        total += fractions[name]

    # Caclulate element fraction of each peak
    for name in names:
        fractions[name] /= total

    return fractions


def fraction_components(
    fractions: np.ndarray, bins: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Binned ratios for element fractions.
    Calculates the mean value and count of peaks with a combination of elements separated by `bins`.
    This can be used to find the different elemental composition of NPs in a sample.

    Args:
        fractions: ratios returned by `detection_element_fractions`
        bins: bins for each element fraction, defaults to [0.0, 0.1, ... 1.0]

    Returns:
        mean of each combination
        count of each combination"""
    if bins is None:
        bins = np.linspace(0, 0.9, 10)

    # Generate indicies from histogram of each element
    hist = np.stack(
        [np.digitize(fractions[name], bins=bins) for name in fractions.dtype.names],
        axis=1,
    )
    # Unique combinations across all histogram indicies
    _, idx, counts = np.unique(hist, axis=0, return_inverse=True, return_counts=True)

    # Calculate the mean value for each unique combination
    compositions = np.empty(counts.size, dtype=fractions.dtype)
    for name in fractions.dtype.names:
        compositions[name] = np.bincount(idx, fractions[name]) / counts

    return compositions, counts


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
    c (mol/L) = m (kg) / (V[4.0 / 3.0 * pi * (d (m) / 2) ^ 3] (m^3) * 1000 (L/m^3)) / M (kg/mol)

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

    return (mass * count) / (flowrate * time * concentration)


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
    return (mass * response_factor * mass_fraction) / (signal * (dwell * flowrate))


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
    """Number concentratioe total volume of the industrial chemical introduced in a registration year bythe person does not excn of particles.
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
