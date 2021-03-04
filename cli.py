import sys
import argparse
import numpy as np
from pathlib import Path

import nanopart

from typing import List, Tuple


def parse_argv(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Acquisiton file
    parser.add_argument("aquisition", help="Output with 2 columns, Time and Counts.")

    # Acquisiton modifiers
    acqgrp = parser.add_argument_group("acquisition arguments")
    acqgrp.add_argument(
        "--cps",
        action="store_true",
        help="Convert acquistion and response signal from CPS to counts.",
    )
    acqgrp.add_argument(
        "--dwelltime",
        metavar="s",
        type=float,
        help="Dwell time in seconds. "
        "The default value is determined using the acquistion file.",
    )
    acqgrp.add_argument(
        "--trim",
        metavar=("Start", "End"),
        type=int,
        nargs=2,
        help="Number of points to trim from start and end of acquistion.",
    )

    # Required inputs
    reqgrp = parser.add_argument_group("required arguments")
    reqgrp.add_argument(
        "--density",
        metavar="g/cm3",
        type=float,
        required=True,
        help="Particle material density.",
    )
    reqgrp.add_argument(
        "--efficiency",
        metavar="[0 - 1]",
        type=float,
        help="Nebulisation efficiency.",
        required=True,
    )
    reqgrp.add_argument(
        "--flowrate",
        metavar="ml/min",
        type=float,
        help="Sample flow rate.",
        required=True,
    )
    reqgrp.add_argument(
        "--response",
        metavar="μg/L",
        type=float,
        help="Response factor in counts per ppb.",
        required=True,
    )

    # Optional arugments
    parser.add_argument(
        "--molarmass",
        metavar="g/mol",
        type=float,
        help="Molecular weight of particle material.",
    )
    parser.add_argument(
        "--massfraction",
        type=float,
        help="Molar ratio between unit cell and analyte.",
        default=1.0,
    )
    # parser.add_argument(
    #     "--responsecps",
    #     action="store_true",
    #     help="Response given in CPS instead of counts.",
    # )

    # Ouptput
    parser.add_argument("--output", type=Path, help="Output data to a text file.")

    args = parser.parse_args(argv[1:])

    # Ensure efficiency is 0.0 to 1.0.
    if args.efficiency is not None and not (0.0 <= args.efficiency <= 1.0):
        parser.error("--efficiency must be a value from 0.0 to 1.0.")

    return args


def rescale_prefix(x: float, prefix: str) -> Tuple[float, str]:
    units = {
        "f": 1e-15,
        "p": 1e-12,
        "n": 1e-9,
        "μ": 1e-6,
        "m": 1e-3,
        "": 1.0,
        "k": 1e3,
        "M": 1e9,
        "G": 1e12,
    }
    base = x * units[prefix]

    prefixes = list(units.keys())
    factors = list(units.values())
    idx = np.max(np.searchsorted(factors, base) - 1, 0)

    return base / factors[idx], prefixes[idx]


if __name__ == "__main__":
    args = parse_argv(sys.argv)
    x, y = np.genfromtxt(
        args.aquisition, delimiter=",", skip_header=4, skip_footer=4, unpack=True
    )

    if not args.dwelltime:
        # Read dwell time (to nearest μs) from the aquisition
        args.dwelltime = np.around(np.mean(np.diff(x)), decimals=6)

    # Convert from counts per second to counts
    if args.cps:
        y = y * args.dwelltime
        args.response = args.response * args.dwelltime

    # Trim data
    if args.trim:
        x = x[args.trim[0] : x.size - args.trim[1]]
        y = y[args.trim[0] : y.size - args.trim[1]]

    # Mean of background
    ub = np.mean(y)
    # Gross critical values
    yc, yd = nanopart.poisson_limits(ub)
    # Critical values
    lc, ld = yc + ub, yd + ub

    detections, regions = nanopart.accumulate_detections(y - ub, yc, yd)

    # Calculate background mean of non-detections
    ndub = np.mean(y[regions == 0])

    # Inputs, converted into kg, L, m, s
    time = x[-1] - x[0]  # s
    flowrate = args.flowrate * 1e-3 / 60.0  # ml/min -> L/s
    response = args.response * 1e9  # L/μg -> L/kg
    density = args.density * 1e-3 * 1e6  # g/cm3 -> kg/m3

    # Nebulisation efficiency
    # if args.efficiency is None:
    #     ref_conc = args.conc * 1e-18  # fg/L -> kg/L
    #     ref_mass = args.mass * 1e-18  # fg -> kg
    #     args.efficiency = nebulisation_efficiency(
    #         count, ref_conc, ref_mass, flowrate, time
    #     )
    #     assert 0.0 <= args.efficiency <= 1.0

    # Particle calculations
    masses = nanopart.particle_mass(
        detections,
        dwell=args.dwelltime,
        efficiency=args.efficiency,
        flowrate=flowrate,
        response_factor=response,
        mass_fraction=args.massfraction,
    )
    sizes = nanopart.particle_size(masses, density=density)
    number = nanopart.particle_number_concentration(
        detections.size, efficiency=args.efficiency, flowrate=flowrate, time=time
    )
    conc = nanopart.particle_total_concentration(
        masses, efficiency=args.efficiency, flowrate=flowrate, time=time
    )
    ionic = ndub / response  # kg/L

    # Calculate number of atoms if molarmass provided
    if args.molarmass:
        molarmass = args.molarmass * 1e-3  # g/mol -> kg/mol
        atoms = nanopart.particle_number_atoms(masses, molarmass=molarmass)

    # Background equivalent calculations
    bemass = nanopart.particle_mass(
        ndub,
        dwell=args.dwelltime,
        efficiency=args.efficiency,
        flowrate=flowrate,
        response_factor=response,
        mass_fraction=args.massfraction,
    )
    # belc = particle_mass(
    #     lc, args.dwelltime, args.efficiency, flowrate, response, args.massfraction
    # )
    beld_size = nanopart.particle_size(
        nanopart.particle_mass(
            ld, args.dwelltime, args.efficiency, flowrate, response, args.massfraction
        ),
        density,
    )
    # Calculate BE atoms if molarmass provided
    if args.molarmass:
        beatoms = nanopart.particle_number_atoms(bemass, molarmass)

    # Convert required from printing
    # conc = conc * 1e18 * 1e-3  # kg/L -> fg/ml
    # number = number * 1e-3  # /L -> /ml
    # ionic = ionic * 1e18 * 1e-3  # kg/L -> fg/ml
    # mean_size = np.mean(sizes) * 1e-6  # m -> μm
    # median_size = np.median(sizes) * 1e-6  # m -> μm
    # beld_size = beld_size * 1e-6  # m -> μm

    conc, conc_pre = rescale_prefix(conc, "k")
    ionic, ionic_pre = rescale_prefix(ionic, "k")
    mean, mean_pre = rescale_prefix(np.mean(sizes), "")
    median, median_pre = rescale_prefix(np.median(sizes), "")
    beld, beld_pre = rescale_prefix(beld_size, "")

    text = (
        f"Detected particles {detections.size}\n"
        f"Number concentration: {number} /L\n"
        f"Concentration: {conc} {conc_pre}g/L\n"
        f"Ionic background concentration: {ionic} {ionic_pre}g/L\n"
        f"Mean NP size: {mean} {mean_pre}m\n"
        f"Median NP size: {median} {median_pre}m\n"
        f"LOD equivalent size: {beld} {beld_pre}m\n"
    )

    if args.molarmass:
        text += (
            f"Median atoms per particle: {np.median(atoms)}\n"
            f"Background equivalent atoms: {beatoms}\n"
        )

    # Output
    if args.output:
        if args.molarmass:
            header = text + "Masses,Sizes,Atoms"
            data = np.stack((masses, sizes, atoms), axis=1)
        else:
            header = text + "Masses,Sizes"
            data = np.stack((masses, sizes), axis=1)

        np.savetxt(
            args.output,
            data,
            delimiter=",",
            header=header,
        )
    else:  # Print to stdout
        print(text)
