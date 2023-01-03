import argparse
from pathlib import Path
from typing import List

import numpy as np
import numpy.lib.recfunctions as rfn


def make_elements(
    path: str, delimiter: str = "\t", drop_names: List[str] = ["MP", "BP", "Ref"]
):
    elements = np.genfromtxt(
        path,
        delimiter=delimiter,
        skip_header=1,
        dtype=[
            ("Number", np.uint16),
            ("Symbol", "U2"),
            ("Name", "U16"),
            ("MW", np.float32),
            ("Density", np.float32),
            ("MP", np.float32),
            ("BP", np.float32),
            ("Ref", np.uint8),
        ],
    )
    elements["MP"] -= 273.15  # Convert to C
    elements["BP"] -= 273.15  # Convert to C
    return rfn.drop_fields(elements, drop_names)


def make_isotopes(path: str, delimiter: str = "\t", drop_names: List[str] = ["Ref"]):
    isotopes = np.genfromtxt(
        path,
        delimiter=delimiter,
        skip_header=1,
        dtype=[
            ("Number", np.uint16),
            ("Symbol", "U2"),
            ("Isotope", np.uint16),
            ("Mass", np.float32),
            ("Composition", np.float32),
            ("Preferred", np.uint8),
            ("Ref", np.uint8),
        ],
    )
    return rfn.drop_fields(isotopes, drop_names)


def make_inorganic(path: str, delimiter: str = "\t", drop_names: List[str] = ["Ref"]):
    inorganic = np.genfromtxt(
        path,
        delimiter=delimiter,
        skip_header=1,
        dtype=[
            ("Formula", "U16"),
            ("Name", "U64"),
            ("CAS", "U12"),
            ("Density", np.float32),
            ("Ref", np.uint8),
        ],
    )
    return rfn.drop_fields(inorganic, drop_names)


def make_polymer(path: str, delimiter: str = "\t", drop_names: List[str] = ["Ref"]):
    polymer = np.genfromtxt(
        path,
        delimiter=delimiter,
        skip_header=1,
        dtype=[
            ("Formula", "U16"),
            ("Name", "U64"),
            ("CAS", "U12"),
            ("Density", np.float32),
            ("Ref", np.uint8),
        ],
    )
    return rfn.drop_fields(polymer, drop_names)


def make_refs(path: str, delimiter: str = "\t"):
    return np.genfromtxt(
        path,
        delimiter=delimiter,
        skip_header=1,
        dtype=[
            ("Ref", np.uint8),
            ("Source", "U64"),
            ("ID", "U64"),
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=Path, help="Path to data directory.")
    args = parser.parse_args()

    np.savez_compressed(
        "../spcal/resources/npdb.npz",
        elements=make_elements(args.dir.joinpath("elements.csv")),
        isotopes=make_isotopes(args.dir.joinpath("isotopes.csv")),
        inorganic=make_inorganic(args.dir.joinpath("inorganic.csv")),
        polymer=make_polymer(args.dir.joinpath("polymer.csv")),
        # refs=make_refs(args.dir.joinpath("refs.csv")),
    )
