import numpy as np

elements = np.genfromtxt(
    "data/elements.csv",
    delimiter="\t",
    skip_header=1,
    dtype=[
        ("Number", int),
        ("Symbol", "U2"),
        ("Name", "U16"),
        ("MW", float),
        ("Density", float),
        ("MP", float),
        ("BP", float),
        ("Ref", int),
    ],
)
elements["MP"] -= 273.15  # Convert to C
elements["BP"] -= 273.15  # Convert to C

inorganic = np.genfromtxt(
    "data/inorganic.csv",
    delimiter="\t",
    skip_header=1,
    usecols=(0, 1, 2, 3, 4),
    dtype=[
        ("Formula", "U16"),
        ("Name", "U64"),
        ("CAS", "U12"),
        ("Density", float),
        ("Ref", int),
    ],
)

polymer = np.genfromtxt(
    "data/polymer.csv",
    delimiter="\t",
    skip_header=1,
    usecols=(0, 1, 2, 3, 4),
    dtype=[
        ("Formula", "U16"),
        ("Name", "U64"),
        ("CAS", "U12"),
        ("Density", float),
        ("Ref", int),
    ],
)

refs = np.genfromtxt(
    "data/refs.csv",
    delimiter="\t",
    skip_header=1,
    dtype=[
        ("Ref", int),
        ("Source", "U64"),
        ("ID", "U32"),
    ],
)

np.savez_compressed(
    "../spcal/resources/npdb.npz",
    elements=elements,
    inorganic=inorganic,
    polymer=polymer,
    refs=refs,
)

