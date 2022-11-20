import numpy as np

elements = np.genfromtxt(
    "data/elements.csv",
    delimiter="\t",
    skip_header=1,
    dtype=[
        ("number", int),
        ("symbol", "U2"),
        ("name", "U16"),
        ("mw", float),
        ("density", float),
        ("mp", float),
        ("bp", float),
        ("ref", int),
    ],
)

inorganic = np.genfromtxt(
    "data/inorganic.csv",
    delimiter="\t",
    skip_header=1,
    usecols=(0,1,2,3,4),
    dtype=[
        ("formula", "U16"),
        ("name", "U64"),
        ("cas", "U12"),
        ("density", float),
        ("ref", int),
    ],
)

polymer = np.genfromtxt(
    "data/polymer.csv",
    delimiter="\t",
    skip_header=1,
    usecols=(0,1,2,3,4),
    dtype=[
        ("formula", "U16"),
        ("name", "U64"),
        ("cas", "U12"),
        ("density", float),
        ("ref", int),
    ],
)

refs = np.genfromtxt(
    "data/refs.csv",
    delimiter="\t",
    skip_header=1,
    dtype=[
        ("ref", int),
        ("source", "U64"),
        ("id", "U32"),
    ],
)

np.savez_compressed("../spcal/resources/npdb.npz", elements=elements, inorganic=inorganic, polymer=polymer, refs=refs)
